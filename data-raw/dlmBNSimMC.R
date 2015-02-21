# dlmBNSimMC.R Hierarchical simulation of BN model for MC estimation

# include functions
rm(list = ls())

library(dlm)
library(LearnBayes)
library(Matrix)
library(trustOptim)

set.seed(10503)
# set.seed(105)

include.phi <- FALSE
include.c <- TRUE
include.u <- TRUE

rmvMN <- function(ndraws, M = rep(0, nrow(S) * ncol(C)), C, S) {
    ## set.seed(153)
    L <- chol(S) %x% chol(C)
    z <- rnorm(length(M))
    if (length(M) == 1) {
        res <- matrix(t(L) %*% z, nrow = nrow(C), ncol = ncol(S))
    }  else {
        res <- matrix(as.vector(M) + t(L) %*% z, nrow = nrow(C), ncol = ncol(S))
    }
    return(res)
}

# create Y
N <- 22  # number of 'sites'
T <- 150  # number of time periods
Tb <- 0  # number of burnin periods
J <- 2  # number of equations
P <- J  # number of time varying covariates per city (excluding intercept)
K1 <- 2  # number of non time varying covariates per city at top level (including intercept)
##Cvec <- rnorm(J, mean = 0.1, sd = 0.03)  # wearout per period
##Uvec <- rnorm(J, mean = 0.15, sd = 0.05)  # wearout due to repetition

Cvec <- seq(.1,.5,length=J)
Uvec <- seq(.8, .03, length=J)

## parameters
delta <- 0.15
Theta12 <- rnorm(K1 * J, sd = 0.3)
dim(Theta12) <- c(K1, J)
V <- list()
FF <- list()
JFF <- list()

# advertising covariates
A <- (matrix(runif(J * T), nr = T, nc = J) > 0.5) *
    matrix(runif(J * T, max = exp(9)), nrow = T, ncol = J)  # total 'advertising' across all sites, for each equation 

# scale/center A Ac<-scale(A)
 Ac <- A / 1e+06
##Ac <- log1p(A)

## WHAT IS aA?

## aA <- matrix(0, nr = T, nc = J)
## colnames(aA) <- paste("aA", 1:J, sep = ".")
## for (j in 1:J) {
##     aA[, j] <- 1 - Cvec[j] - Uvec[j] * Ac[,j] - delta * (1-(A[,j] > 0))
## }


gA <- log(1+A)

E <- rpois(T * J, 0.5)  # incidence of new creatives
dim(E) <- c(T, J)
E[A == 0] <- 0  # switch to zero if there is no advertising
phi <- matrix(0, nc = J, nr = J)  # response coefficients for new creatives
diag(phi) <- runif(J, min = 0.05, max = 0.1)

# time invariant component
## K <- 2
K <- K1  ## should we have hard-coded K like that?
F12l <- list()
for (t in 1:T) {
    F12l[[t]] <- rnorm(N * K, sd = 0.5)
    dim(F12l[[t]]) <- c(N, K)
}

# time varying component
F1l <- F1ml <- list()
for (t in 1:T) {
    F1l[[t]] <- matrix(0, nrow = N, ncol = N * (1 + P))
    F1ml[[t]] <- Matrix(0, nrow = N, ncol = N * (1 + P))
    for (i in 1:N) {
        .draw <- c(1, log(rexp(P)))
        F1l[[t]][i, ((i-1) * (P+1) + 1):((i-1) * (P+1)+P+1)] <- .draw
        F1ml[[t]][i, ((i-1) * (P+1) + 1):((i-1) * (P+1)+P+1)] <- .draw
    }
    F1ml[[t]] <- t(as(F1ml[[t]], "dgCMatrix"))  ## make sparse
}

W <- .1 * diag(1 + J + P)
# W[1]<-0.01 diag(W)[(2+J):(1+J+P)]<-0.001
Sigma <- matrix(0, nrow = J, ncol = J)
diag(Sigma) <- 0.1

V[[1]] <- diag(.1, nrow = N)
## for(i in 1:(ncol(V[[1]])-1)) V[[1]][i,i+1]<-V[[1]][i+1,i] <- 0.02

V[[2]] <- diag(N * (1 + P)) * 0.1

Theta2.0 <- matrix(rep(0, 1 + J + P), nrow = (1 + J + P), ncol = J)
Theta2.0[1, ] <- 25
Theta2.0[1 + 1:J, ] <- -0.005
# Theta2.0[1+1:J,]<- 0
Theta2.0[(J + 2):(1 + J + P), ] <- 1
for (j in 1:J) {
    Theta2.0[1+J+j, j] <- -2
    Theta2.0[1+j, j] <- 0.25
    # Theta2.0[1+J+j,j]<- 0 Theta2.0[1+j,j]<- 0
}

# true parameters
T1 <- list()
T2 <- list()

##################################################################### simulate data here
Theta2t <- Theta2.0  # initialise
Y <- NULL
Yl <- list()
for (t in 1:T) {
    FF[[1]] <- F1l[[t]]
    
    F2t <- dlm::bdiag(matrix(c(1, rep(0, J)), nc = 1 + J), diag(P))
    for (n in 2:N) {
        F2t <- rbind(F2t,
                     dlm::bdiag(matrix(c(1, rep(0, J)), 
                                       nc = 1 + J), diag(P))
                     )
    }
    FF[[2]] <- F2t
    
    Gt <- diag(1 + J + P)
    Gt[1,1] <- (1 - delta)
    Gt[1, 1 + (1:J)] <- gA[t, ]
    for (j in 1:J) {
        if (include.c) {
            if (include.u) {
                Gt[j+1, j+1] <- exp(-Cvec[j] - Ac[t,j] * log(Uvec[j]))              
            } else {
                Gt[j+1, j+1] <- exp(-Cvec[j])
            }
        } else {
            if (include.u) {
##                Gt[j+1, j+1] <- exp(-Uvec[j]*Ac[t,j])
                Gt[j+1, j+1] <- exp(-Ac[t,j] * log(Uvec[j]))              
            } else {
                Gt[j+1, j+1] <- 1
            }
        }
    }
        
    ## innovation component which switches signs if the underlying state
    ## variable does (simplified here)
    Ht <- matrix(0, nr = J, nc = J)
    for (j in 1:J) {
        Ht[j,] <- delta * (A[t,j]==0)
        if (include.phi) {
            Ht[j,] <- Ht[j,] + phi[j,]*E[t,j]
        }
    }
    
 ##   for (k in 1:J) {
        ## if(Theta2t[1+j,k] < 0) Ht[1+j,k] <- -delta*(1-(A[t,j]>0)) -
        ## Evec[j,k]*E[t,j] # if it is below zero else Ht[j+1,k] <-
        ## delta*(1-(A[t,j]>0)) + Evec[j,k]*E[t,j] # if above if(Theta2t[1+j,k]
        ## < 0) Ht[1+j,k] <- - Evec[j,k]*E[t,j] # if it is below zero else
        ## Ht[j+1,k] <- Evec[j,k]*E[t,j] # if above if(j!=k) Ht[1+j,k] <-
        ## -delta*(1-(A[t,j]>0)) - e[j]*E[t,j] # if it is below zero else
        ## Ht[j+1,k] <- delta*(1-(A[t,j]>0)) + e[j]*E[t,j] # if above
        ##       Ht[j, k] <- Evec[j, k] * E[t, j]
   ## }

    
    epsW <- rmvMN(1, , W, Sigma)
    Theta2t <- Gt %*% Theta2t + epsW
    Theta2t[2:(J+1),] <- Theta2t[2:(J+1),] + Ht
    T2[[t]] <- Theta2t
    
    epsV2 <- rmvMN(1, , V[[2]], Sigma)
    Theta1t <- FF[[2]] %*% Theta2t + epsV2
    T1[[t]] <- Theta1t
    epsV1 <- rmvMN(1, , V[[1]], Sigma)
    Yt <- FF[[1]] %*% Theta1t + F12l[[t]] %*% Theta12 + epsV1

    Y <- rbind(Y, as.vector(Yt))
    Yl[[t]] <- Yt
}

Y <- Y[-(1:Tb), ]

T1true <- list()
T2true <- list()
F12 <- list()
F1m <- list()
F1 <- list()
Y <- list()

for (i in (Tb + 1):T) {
    T1true[[i - Tb]] <- T1[[i]]
    T2true[[i - Tb]] <- T2[[i]]
    F12[[i - Tb]] <- F12l[[i]]
    F1m[[i - Tb]] <- F1ml[[i]]
    F1[[i - Tb]] <- F1l[[i]]
    Y[[i - Tb]] <- Yl[[i]]
}

if (Tb > 0) { ## if there is burnin
    gA <- gA[-(1:Tb), ]
    ## aA <- aA[-(1:Tb), ]
    A <- A[-(1:Tb), ]
    Ac <- Ac[-(1:Tb), ]
    T <- (T - Tb)
    E <- E[-(1:Tb), ]
}

Al <- El <- list()
for (t in 1:T) {
    El[[t]] <- E[t, ]
    Al[[t]] <- A[t, ]
}

### convert
dimensions <- list(N = N, T = T, J = J, K = K, P = P)

## F1 - covariates with time-varying coefficients
F2 <- list()
b1 <- Matrix(c(1, rep(0, J + P)), nrow = 1, ncol = 1 + J + P)
b2 <- Matrix(0, nrow = P, ncol = J + 1)
b3 <- Diagonal(P)
B <- rBind(b1, cBind(b2, b3))

for (t in 1:T) {
    F2[[t]] <- kronecker(Matrix(rep(1, N), nrow = N, ncol = 1), B)
    F2[[t]] <- t(as(F2[[t]], "dgCMatrix"))
}

mcmod <- list(dimensions = dimensions, Y = Y, E = El, A = Al, X = F12, 
    F1 = F1m, F2 = F2)

truevals <- list(T1=T1true, T2=T2true, Theta12=Theta12, V=V, Sigma=Sigma, W=W,
                 Cvec=Cvec, Uvec=Uvec)

### saving for mcmod object
save(mcmod, truevals, file = "./data/mcmodsim.RData")
