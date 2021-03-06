rm(list=ls())
gc()

library(Matrix)
library(Rcpp)
library(RcppEigen)
library(numDeriv)
library(trustOptim)
library(plyr)
library(reshape2)
library(ads)

set.seed(1234)

start.true.pars <- FALSE

mod.name <- "hdlm"
data.name <- "tti"
##data.file <- paste0("~/Documents/hdlm/data/mcmod",data.name,".Rdata")
save.file <- paste0("~/Documents/hdlm/results/",mod.name,"_",data.name,"_mode.Rdata")


flags <- list(include.H=FALSE,
              add.prior=TRUE,
              include.X=TRUE,
              standardize=FALSE,
              A.scale = 1000000
              )

nfact.V1 <- 0
nfact.V2 <- 1
nfact.W2 <- 1


get.f <- function(P, ...) return(cl$get.f(P))
get.df <- function(P, ...) return(cl$get.fdf(P)$grad)
get.hessian <- function(P, ...) return(cl$get.hessian(P))
get.f.direct <- function(P, ...) return(cl$get.f.direct(P))
get.LL <- function(P, ...) return(cl$get.f.direct(P))
get.hyperprior <- function(P, ...) return(cl$get.hyperprior(P))

##load(data.file)
print("hello")
dn <- paste0("mcmod",data.name) 
data(list=dn)
mcmod <- eval(parse(text=dn))


N <- mcmod$dimensions$N
T <- mcmod$dimensions$T
J <- mcmod$dimensions$J

if (flags$include.X) K <- mcmod$dimensions$K else K <- 0
P <- mcmod$dimensions$P

Y <- mcmod$Y[1:T]
if (flags$standardize) {
  Y.mean <- mean(acast(melt(Y),Var1~Var2,fun.aggregate=mean))
  Y.sd <- mean(acast(melt(Y),Var1~Var2,fun.aggregate=sd))
  Y <- llply(Y,scale,center=rep(Y.mean,J),scale=rep(Y.sd,J))
}

F1 <- mcmod$F1[1:T]
F2 <- mcmod$F2[1:T]
A <- mcmod$A[1:T]
##A <- llply(mcmod$A[1:T],function(x) return(x/flags$A.scale))

if (flags$include.H) {
  E <- mcmod$E[1:T]
} else {
  E <- NULL
}

if (flags$include.X) {
  X <- mcmod$X[1:T]
  if (flags$standardize) {
    X.mean <- colMeans(acast(melt(X),Var1~Var2,fun.aggregate=mean))
    X.sd <- colMeans(acast(melt(X),Var1~Var2,fun.aggregate=sd))
    X <- llply(X,scale,center=X.mean, scale=X.sd)
  }
} else {
  if (K!=0) stop("If not including X, K must be zero.")
  X <- NULL
}

## note:  F1 and F2 are transposed from what is in the note.
## F1 is N(1+P) x N; F2 is (1+P+J) x N(1+P)
## these matrices are transposed when passed to the C++
## code, to force row-major order.

data <- list(X=X, Y=Y, F1=F1, F2=F2,
             A=A, E=E)


dim.V1 <- N
dim.V2 <- N*(P+1)
dim.W1 <- J+1
dim.W2 <- P
dim.W <- dim.W1+dim.W2

dimensions <- c(N=N,T=T,J=J,K=K,P=P,
                nfact.V1=nfact.V1,
                nfact.V2=nfact.V2,
                nfact.W2=nfact.W2
                )

max.nfact.V1 <- 1+2*dim.V1-sqrt(1+8*dim.V1)
max.nfact.V2 <- 1+2*dim.V2-sqrt(1+8*dim.V2)
max.nfact.W2 <- 1+2*dim.V2-sqrt(1+8*dim.W2)

if (nfact.V1 > max.nfact.V1) stop("Too many factors for V1")
if (nfact.V2 > max.nfact.V2) stop("Too many factors for V2")
if (nfact.W2 > max.nfact.W2) stop("Too many factors for W2")


## priors

## We have to include these prior parameters

M20 <- matrix(0,1+P+J,J)
## M20[1,] <- 15
## M20[2:(J+1),] <- -.005
## M20[(J+2):(1+P+J),] <- 1
## for (j in 1:J) {
##     M20[J+1+j,j] <- -2
##     M20[j+1,j] <- .25
## }
C20 <- 500*diag(1+P+J,1+P+J)

E.Sigma <- 0.1 * diag(J) ## expected covariance across brands
nu0 <- P + 2*J + 6  ## must be greater than theta2 rows+cols
Omega0 <- (nu0-J-1)*E.Sigma

## The following priors are optional

if (flags$add.prior) {

   
  if (flags$include.H) {
    ## prior on phi:  matrix normal with sparse covariances
    mean.phi <- matrix(0,J,J)
    cov.row.phi <- 10*diag(J)
    cov.col.phi <- 10*diag(J)
    chol.cov.row.phi <- t(chol(cov.row.phi))
    chol.cov.col.phi <- t(chol(cov.col.phi))
    
    prior.phi <- list(mean=mean.phi,      
                      chol.row = chol.cov.row.phi,      
                      chol.col = chol.cov.col.phi
                      )
  } else {
    prior.phi <- NULL
  }

  if (flags$include.X) {
    ## prior on theta12:  matrix normal with sparse covariances
    mean.theta12 <- matrix(0,K,J)
    cov.row.theta12 <- 500*diag(K) ## across covariates within brand
    cov.col.theta12 <- 500*diag(J) ## across brand within covariates
    chol.cov.row.theta12 <- t(chol(cov.row.theta12))
    chol.cov.col.theta12 <- t(chol(cov.col.theta12))
    
    prior.theta12 <- list(mean=mean.theta12,
                          chol.row = chol.cov.row.theta12,
                          chol.col = chol.cov.col.theta12
                          )
  } else {
    prior.theta12 <- NULL
  }
  
  ## prior on logit.delta.  transformed beta with 2 parameters
  
##  prior.delta <- list(a=1,b=5)
  prior.delta <- list(a=1,b=1)
  
  ## prior on c.mean and u.mean:  normal
  ## prior on c.sd and u.sd:  truncated normal

  prior.c.mean <- 0
  
  
  prior.c <- list(mean.mean=prior.c.mean,
                  mean.sd=1,
                  sd.mean=.1,
                  sd.sd=.5)

  prior.u.mean <- 0           
                 
  prior.u <- list(mean.mean=prior.u.mean,
                  mean.sd=1,
                  sd.mean=.1,
                  sd.sd=.5)
  
  ## For V1, V2, and W2:   T or half-T priors
  
  prior.V1 <- list(diag.scale=.01, diag.df=4,
                   fact.scale=.01, fact.df=4)
  prior.V2 <- list(diag.scale=.01, diag.df=4,
                   fact.scale=.01, fact.df=4)
  prior.W2 <- list(diag.scale=.01, diag.df=4,
                   fact.scale=.01, fact.df=4)

  ## LKJ prior on W1.  Scale parameter has half-T prior
  
  prior.W1 <- list(scale.df=4, scale.s=1, eta=1)
  
} else {
  
  prior.phi <- NULL
  prior.V1 <- NULL
  prior.V2 <- NULL
  prior.W1 <- NULL
  prior.W2 <- NULL
  prior.delta <- NULL
  prior.c <- NULL
  prior.u <- NULL
  prior.theta12 <- NULL
}
  
tmp <- list(M20=M20,
            C20=C20,
            Omega0=Omega0,
            nu0=nu0,
            phi=prior.phi,
            theta12=prior.theta12,
            delta=prior.delta,
            c=prior.c,
            u=prior.u,
            V1=prior.V1, V2=prior.V2,
            W1=prior.W1, W2=prior.W2
            )

priors <- Filter(function(x) !is.null(x), tmp)

DL <- list(data=data, priors=priors,
           dimensions=dimensions,
           flags=flags)

## starting parameters
if (flags$include.X) {
    if (start.true.pars) {
        theta12.start <- true.pars$theta12
    } else {
        theta12.start <- matrix(0,K,J)
    }
} else {
  theta12.start <- NULL
}

if (start.true.pars) {
    c.mean.log.sd.start <- true.pars$c.mean.log.sd
    c.off.start <- true.pars$c.off
    u.mean.log.sd.start <- true.pars$u.mean.log.sd
    u.off.start <- true.pars$u.off
    logit.delta.start <- true.pars$logit.delta   
} else {
    c.mean.log.sd.start <- c(0,0)
    c.off.start <- rep(0,J)
    u.mean.log.sd.start <- c(0,0)
    u.off.start <- rep(0,J)
    logit.delta.start <- 0
}


if (flags$include.H) {
    phi.start <- matrix(0,J,J)
} else {
    phi.start <- NULL
}


V1.length <- N + N*nfact.V1 - nfact.V1*(nfact.V1-1)/2
V2.length <- N*(P+1) + N*(P+1)*nfact.V2 - nfact.V2*(nfact.V2-1)/2
W1.length <- dim.W1*(dim.W1-1)/2 + 1
W2.length <- P + P*nfact.W2 - nfact.W2*(nfact.W2-1)/2

##V1.start <- (1:V1.length)/10
##V2.start <- (1:V2.length)/20
##W1.start <- (1:W1.length)/20
##W2.start <- (2:(W2.length+1))/30

 ## V1.start <- rnorm(V1.length) - 3
 ## V2.start <- rnorm(V2.length) - 3
 ## W1.start <- rnorm(W1.length) - 3
 ## W2.start <- rnorm(W2.length) - 3

 V1.start <- rep(0,V1.length)
 V2.start <- rep(0,V2.length)
 W1.start <- rep(0,W1.length)
 W2.start <- rep(0,W2.length)

tmp <- list(
    theta12=theta12.start,
    c.mean.log.sd=c.mean.log.sd.start,
    c.off=c.off.start,
    u.mean.log.sd=u.mean.log.sd.start,
    u.off=u.off.start,
    phi=phi.start,
    logit.delta=logit.delta.start,
    V1=V1.start,
    V2=V2.start,
    W1=W1.start,
    W2=W2.start
    )

start.list <- as.relistable(Filter(function(x) !is.null(x), tmp))

start <- unlist(start.list)


cat("Setting up\n")
tset <- system.time(cl <- new("ads", DL))
print(tset)

cat("Recording tape\n")
trec <- system.time(cl$record.tape(start))

cat("Objective function - taped\n")
tmp <- get.f(start)
tf <- system.time(f <- get.f(start))
cat("f = ",f,"\n")
print(tf)

cat("gradient\n")
tg <- system.time(df <- get.df(start))
print(tg)

stop()

opt2 <- optim(start,
             fn=get.f,
             gr=get.df,
             hessian=FALSE,
             method="BFGS",
             control=list(
               fnscale=-1,
               REPORT=1,
               trace=3,
               maxit=1000
               )
             )

opt3 <- trust.optim(opt2$par,                    
                    fn=get.f,
                    gr=get.df,       
                    method="SR1",
                    control=list(
                        report.level=5L,
                        report.precision=3L,
                        maxit=3000L,
                        function.scale.factor=-1,
                        preconditioner=0,
                        stop.trust.radius=1e-12,
                        contract.factor=.4,
                        expand.factor=2,
                        expand.threshold.radius=.85
                        )
                    )


tmp <- trust.optim(opt3$solution,
                   fn=get.f,
                   gr=get.df,
                   method="BFGS",
                   control=list(
                       report.level=5L,
                       report.precision=3L,
                       maxit=3000L,
                       function.scale.factor=-1,
                       preconditioner=0,
                       stop.trust.radius=1e-12,
                       contract.factor=.7
                     )
                   )

opt <- trust.optim(tmp$solution,
                   fn=get.f,
                   gr=get.df,
                   method="BFGS",
                   control=list(
                       report.level=5L,
                       report.precision=3L,
                       maxit=3000L,
                       function.scale.factor=-1,
                       preconditioner=0,
                       stop.trust.radius=1e-12,
                       contract.factor=.7
                     )
                   )




opt$par <- opt$solution

                     
sol.vec <- relist(opt$par, skeleton=start.list)

recover.cov.mat <- function(v, d, nfact) {

  ## v:  vector of elements that define the matrix
  ## d:  dimension of the square covariance matrix
  ## nfact:  number of factors that were estimated

  S <- diag(exp(v[1:d]))
  if (nfact>0) {
    F <- matrix(0,nrow=d,ncol=nfact)    
    ind <- d+1
    for (j in 1:nfact) {
      F[j:d,j] <- v[ind:(ind+d-j)]
      F[j,j] <- exp(F[j,j])
      ind <- ind+d-j+1
    }
    S <- S + tcrossprod(F)
  }
  return(S)
}

recover.corr.mat <- function(v, d) {

    ind <- 1
    a <- exp(v[ind])
    ind <- ind+1
    Z <- matrix(0,d,d)
    Z[2:d,1] <- v[ind:(ind+d-1-1)]
    ind <- ind+d-1
    for (j in 2:(d-1)) {
        Z[(j+1):d,j] <- v[ind:(ind+d-j-1)]
        ind <- ind+d-j
    }
    Z <- tanh(Z)

    W <- matrix(0,d,d)
    W[1,1] <- 1
    W[2:d,1] <- Z[2:d,1]
    for (j in 2:d) {
        W[j,j] <- prod(sqrt(1-Z[j,1:(j-1)]^2))
        if (j<d) W[(j+1):d,j] <- W[j,j]*Z[(j+1):d,j]
    }
    X <- a*tcrossprod(W)
    return(X)           
}
 
sol <- sol.vec
sol$delta <- exp(sol$logit.delta)/(1+exp(sol$logit.delta))
sol$cj <- exp(sol$c.mean.log.sd[2]) * sol$c.off + sol$c.mean.log.sd[1]
sol$uj <- exp(sol$u.mean.log.sd[2]) * sol$u.off + sol$u.mean.log.sd[1]
sol$V1 <- recover.cov.mat(sol.vec$V1,dim.V1,nfact.V1)
sol$V2 <- recover.cov.mat(sol.vec$V2,dim.V2,nfact.V2)
sol$W2 <- recover.cov.mat(sol.vec$W2,dim.W2,nfact.W2)
sol$W1 <- recover.corr.mat(sol.vec$W1,dim.W1)

hs <- get.hessian(opt$par)
cv <- solve(-hs)
se <- sqrt(diag(cv))
se.sol <- relist(se,skeleton=start.list)

save(sol, se.sol, opt, dimensions, priors, flags, file=save.file)
