library(rstan)

data.name <- "ptw"      # choose from ptw, tti, lld, dpp for now
stan.code <-"stan3"     # this file located in the inst/script directory and must have .stan suffix
numiter <- 1.0e5            # with diagonal V, this should converge in around 500, comfortably
numcores <- 2           # parallel processing will be done automatically if this is more than one

dn <- paste0("mcmod",data.name) ## name of data file, e.g., mcmoddpp
data(list=dn)  ## load data
mcmod <- eval(parse(text=dn)) ## rename to mcmod

sampler <- "vb"

save.file <- paste0("inst/results/",stan.code,"_", data.name,"_stan_",sampler,".Rdata")

Escale <- 100
N <- mcmod$dimensions$N
T <- as.integer(mcmod$dimensions$T)
J <- mcmod$dimensions$J
Jb <- mcmod$dimensions$Jb
JbE <- mcmod$dimensions$JbE
K <- mcmod$dimensions$K
P <- mcmod$dimensions$P
Yr <- mcmod$Y[1:T]
F1r <- mcmod$F1[1:T]
F2r <- mcmod$F2[1:T]
Ar <- mcmod$A[1:T]
Er <- mcmod$E[1:T]     # choose from E, Ef, and Efl1 for dummy, frac of budget, and frac of budget of lagged dummy
Xr <- mcmod$X[1:T]

Y <- array(dim=c(T, N, J))
X <- array(dim=c(T, N, K))
F1F2 <- array(dim=c(T, N, 1+Jb+P))
F1 <- array(dim=c(T, N, N*(1+P)))
F2 <- array(dim=c(T, N*(1+P), (1+Jb+P)))
A <- array(dim=c(T, Jb))
E <- array(dim=c(T, JbE))


## priors
nu0 <- J + 5;
Omega0 <- diag(J);			# This directly/proportionally impacts the estimated covariance values
M20 <- matrix(0,1+P+Jb,J)
M20[1,] <- 15
M20[2:(Jb+1),] <- -.005
M20[(J+2):(1+P+Jb),] <- 1
for (j in 1:J) {
    M20[Jb+1+j,j] <- -2
    M20[j+1,j] <- .25
}
C20 <- 10*diag(1+P+Jb,1+P+Jb)


for (i in 1:T) {

    Y[i,,] <- Yr[[i]]
    X[i,,] <- Xr[[i]]
    A[i,] <- Ar[[i]]
    E[i,] <- (Er[[i]][1:JbE])/Escale
    F1[i,,] <- t(as(F1r[[i]],"matrix"))
    F2[i,,] <- t(as(F2r[[i]],"matrix"))
    F1F2[i,,] <- F1[i,,]%*%F2[i,,]
}

DL <- list(N=N, T=T, J=J, Jb = Jb, JbE = JbE, K=K, P=P,
           Y=Y, X=X, F1=F1, F2=F2, F1F2 = F1F2,
           A=A, E=E,
           nu0=nu0, Omega0=Omega0,
           M20=M20,C20=C20)


st <- stan_model(paste0("inst/scripts/",stan.code,".stan"))
if(sampler=="NUTS") fit <- sampling(st,
                data=DL,
                iter=numiter, cores=numcores)
if(sampler=="vb") fit <- vb(st, data=DL, iter=numiter)
if(sampler=="MLE") fit <- optimizing(st, data=DL, iter=numiter, verbose=TRUE)

save.image(save.file)




