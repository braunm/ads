library(rstan)

data.name <- "ptw"      # choose from ptw, tti, lld, dpp for now
stan.code <-"stan2"     # this file located in the inst/script directory and must have .stan suffix
numiter <- 10            # with diagonal V, this should converge in around 500, comfortably
numcores <- 2           # parallel processing will be done automatically if this is more than one

data.file <- paste0("data/mcmod",data.name,".RData")
save.file <- paste0("inst/results/",stan.code,"_", data.name,"_stan.Rdata")

load(data.file)

N <- mcmod$dimensions$N
T <- as.integer(mcmod$dimensions$T)
J <- mcmod$dimensions$J
K <- mcmod$dimensions$K
P <- mcmod$dimensions$P
Yr <- mcmod$Y[1:T]
F1r <- mcmod$F1[1:T]
F2r <- mcmod$F2[1:T]
Ar <- mcmod$A[1:T]
Er <- mcmod$E[1:T]
Xr <- mcmod$X[1:T]

Y <- array(dim=c(T, N, J))
X <- array(dim=c(T, N, K))
F1F2 <- array(dim=c(T, N, 1+J+P))
F1 <- array(dim=c(T, N, N*(1+P)))
F2 <- array(dim=c(T, N*(1+P), (1+J+P)))
A <- array(dim=c(T, J))
E <- array(dim=c(T, J))

## priors
nu0 <- J + 5;
Omega0 <- diag(J);			# This directly/proportionally impacts the estimated covariance values
M20 <- matrix(0,1+P+J,J)
M20[1,] <- 15
M20[2:(J+1),] <- -.005
M20[(J+2):(1+P+J),] <- 1
for (j in 1:J) {
    M20[J+1+j,j] <- -2
    M20[j+1,j] <- .25
}
C20 <- 50*diag(1+P+J,1+P+J)

for (i in 1:T) {

    Y[i,,] <- Yr[[i]]
    X[i,,] <- Xr[[i]]
    A[i,] <- Ar[[i]]
    E[i,] <- Er[[i]]
    F1[i,,] <- t(as(F1r[[i]],"matrix"))
    F2[i,,] <- t(as(F2r[[i]],"matrix"))        
    F1F2[i,,] <- F1[i,,]%*%F2[i,,]
}

DL <- list(N=N, T=T, J=J, K=K, P=P,
           Y=Y, X=X, F1=F1, F2=F2, F1F2 = F1F2,
           A=A, E=E,
           nu0=nu0, Omega0=Omega0,
           M20=M20,C20=C20)

st <- stan_model(paste0("inst/scripts/",stan.code,".stan"))


fit <- sampling(st,
                data=DL,
                iter=numiter, cores=numcores)

save.image(save.file)



