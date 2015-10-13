library(rstan)


mod.name <- "hdlm"
data.name <- "sim"

##data.file <- paste0("~/Documents/hdlm/ads/data/mcmod",data.name,".RData")
## save.file <- paste0("~/Documents/hdlm/results/",mod.name,"_",data.name,"_mode.Rdata")

data.file <- paste0("data/mcmod",data.name,".RData")
save.file <- paste0("inst/results/",mod.name,"_",data.name,"_stan.Rdata")

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
F1F2 <- array(dim=c(T, 1+J+P, N))
A <- array(dim=c(T, J))
E <- array(dim=c(T, J))

## priors
nu <- J + 5;
Omega0 <- diag(J);
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
    F1F2[i,,] <- as(F2r[[i]] %*% F1r[[i]],"matrix")
}

DL <- list(N=N, T=T, J=J, K=K, P=P,
           Y=Y, X=X, F1F2=F1F2,
           A=A, E=E,
           nu=nu, Omega0=Omega0,
           M20=M20,C20=C20)

st <- stan_model("inst/scripts/stan1.stan")
fit <- sampling(st,
                data=DL,
                iter=2)





