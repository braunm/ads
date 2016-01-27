# dlmBNSimMC.R - Hierarchical simulation of BN model for MC estimation

############################################################################################################################################

# Preliminaries
rm(list = ls())

library(dlm)
#library(LearnBayes)
library(Matrix)

set.seed(10503)
# set.seed(105)

############################################################################################################################################

# flags here

flags <- list(include.phi=TRUE,
    add.prior=TRUE,
    include.X=TRUE,
    standardize=FALSE,
    A.scale = 1,
    fix.V = FALSE,
    fix.W = FALSE,
    W1.LKJ = FALSE
)

############################################################################################################################################


############################################################################################################################################
# parameter section

N   <- 42                                           # number of 'sites'
T   <- 220                                          # number of time periods
Tb  <- 0                                            # number of burnin periods (discard first Tb simulated time periods)
J   <- 3                                            # number of equations
P   <- J                                            # number of time varying covariates per city (excluding intercept)
K1  <- 2                                            # number of non time varying covariates per city at top level (including intercept)
delta <- 0.1                                        # memory decay parameter
Ascale <- 1.0e6                                     # scale down raw advertising (dollars)
phi <- matrix(runif(J*J,max=0.1), nc = J, nr = J)   # response coefficients for new creatives

# covariance matrixes
W <- .001 * diag(1 + J + P)                         # time covariance
# W[1]<-0.01 diag(W)[(2+J):(1+J+P)]<-0.001
Sigma <- matrix(0, nrow = J, ncol = J)              # covariance across columns
diag(Sigma) <- 0.1

V <- list()
V[[1]] <- diag(.1, nrow = N)                        # covariance across rows
## for(i in 1:(ncol(V[[1]])-1)) V[[1]][i,i+1]<-V[[1]][i+1,i] <- 0.02

V[[2]] <- diag(N * (1 + P)) * 0.1                   # covariance of parameters across cities

# Initial value for Theta_20
Theta2.0 <- matrix(rep(0, 1 + J + P), nrow = (1 + J + P), ncol = J)
Theta2.0[1, ] <- c(10, 16, 8)
Theta2.0[1 + 1:J, ] <- -0.005
# Theta2.0[1+1:J,]<- 0
Theta2.0[(J + 2):(1 + J + P), ] <- 1
for (j in 1:J) {
    Theta2.0[1+J+j, j] <- -2
    Theta2.0[1+j, j] <- 0.25
    # Theta2.0[1+J+j,j]<- 0 Theta2.0[1+j,j]<- 0
}


############################################################################################################################################
# end of parameter section



############################################################################################################################################
### Functions needed here (temporary)

#rmvMN <- function(ndraws, M = rep(0, nrow(S) * ncol(C)), C, S) {
#    ## set.seed(153)
#    L <- chol(S) %x% chol(C)
#    z <- rnorm(length(M))
#    if (length(M) == 1) {
#        res <- matrix(t(L) %*% z, nrow = nrow(C), ncol = ncol(S))
#    }  else {
#        res <- matrix(as.vector(M) + t(L) %*% z, nrow = nrow(C), ncol = ncol(S))
#    }
#    return(res)
#}


############################################################################################################################################
# Initialize simulation

# Create containers for data
Theta12 <- rnorm(K1 * J, mean = .02, sd = 0.03)
dim(Theta12) <- c(K1, J)

FF <- list()
JFF <- list()

# advertising covariates
A <- (matrix(runif(J * T), nr = T, nc = J) > 0.5) *
    matrix(runif(J * T, max = exp(9)), nrow = T, ncol = J)  # total 'advertising' across all sites, for each equation

E <- rpois(T * J, 0.5)  # incidence of new creatives
dim(E) <- c(T, J)
E[A == 0] <- 0  # switch to zero if there is no advertising


# Data transformations
# scale/center A Ac<-scale(A)
Ac <- A / flags$A.scale
gA <- log(1+A)

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

    ## innovation component
    Ht <- matrix(0, nr = J, nc = J)
    for (j in 1:J) {
        if (flags$include.phi) {
            Ht[j,] <- Ht[j,] + phi[j,]*E[t,j]
        }
    }

    epsW <- rmvMN(1, , W, Sigma)
    Theta2t <- Gt %*% Theta2t + epsW

    Theta2t[2:(J+1),] <- Theta2t[2:(J+1),] +  Ht
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
dimensions <- list(N = N, T = T, J = J, Jb = J, JbE = J, K = K, P = P)

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

truevals <- list(T1=T1true, T2=T2true, Theta12=Theta12, V=V, Sigma=Sigma, W=W, phi = phi, Theta2.0=Theta2.0)

### saving for mcmod object
save(mcmod, truevals, file = "./data/mcmodsim.RData")
