mod.name <- "hdlm"
data.name <- "tti"

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
F1F2 <- array(dim=c(T, N, 1+J+P))
A <- array(dim=c(T, J))
E <- array(dim=c(T, J))


for (i in 1:T) {

    Y[i,,] <- Yr[[i]]
    X[i,,] <- Xr[[i]]
    A[i,] <- Ar[[i]]
    E[i,] <- Er[[i]]
    F1F2[i,,] <- as(t(F1r[[i]]) %*% t(F2r[[i]]),"matrix")
}

DL <- list(N=N, T=T, J=J, K=K, P=P,
           Y=Y, X=X, F1F2=F1F2,
           A=A, E=E)

st <- stan_model("inst/scripts/stan1.stan")
fit <- sampling(st,
                data=DL,
                iter=2)





