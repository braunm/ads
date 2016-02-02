rm(list=ls())
gc()

library(Matrix)
library(Rcpp)
library(RcppEigen)
library(numDeriv)
library(trustOptim)
library(plyr)
library(reshape2)

set.seed(1234)


data.name <- "ptw"
data.is.sim <- FALSE

dn <- paste0("mcmod",data.name) ## name of data file, e.g., mcmoddpp
data(list=dn)  ## load data
mcmod <- eval(parse(text=dn)) ## rename to mcmod

if (data.is.sim) {
    flags <- mcmod$trueflags
} else {
    flags <- list(full.phi=FALSE, # default is a diagonal phi matrix
                  add.prior=TRUE,
                  include.X=TRUE,
                  standardize=FALSE,
                  A.scale = 1,
                  fix.V = FALSE,
                  fix.W = FALSE,
                  W1.LKJ = FALSE
                  )
}

nfact.V <- 0
nfact.W1 <- 2
nfact.W2 <- 2

get.f <- function(P, ...) return(cl$get.f(P))
get.df <- function(P, ...) return(cl$get.fdf(P)$grad)
get.hessian <- function(P, ...) return(cl$get.hessian(P))
get.f.direct <- function(P, ...) return(cl$get.f.direct(P))
get.LL <- function(P, ...) return(cl$get.f.direct(P))
get.hyperprior <- function(P, ...) return(cl$get.hyperprior(P))

N <- mcmod$dimensions$N
T <- mcmod$dimensions$T
J <- mcmod$dimensions$J
Jb <- mcmod$dimensions$Jb

if (flags$include.X) K <- mcmod$dimensions$K else K <- 0
P <- mcmod$dimensions$P

Y <- mcmod$Y[1:T]
F1 <- mcmod$F1[1:T]
F2 <- mcmod$F2[1:T]
A <- mcmod$A[1:T]
E <- mcmod$E[1:T]

if (flags$include.X) {
  X <- mcmod$X[1:T]
} else {
  if (K!=0) stop("If not including X, K must be zero.")
  X <- NULL
}

## note:  F1 and F2 are transposed from what is in the note.
## F1 is N(1+P) x N; F2 is (1+P+Jb) x N(1+P)
## these matrices are transposed when passed to the C++
## code, to force row-major order.

data <- list(X=X, Y=Y, F1=F1, F2=F2,
             A=A, E=E)

dim.V <- N
dim.W1 <- Jb+1
dim.W2 <- P
dim.W <- dim.W1+dim.W2

dimensions <- c(N=N,T=T,J=J,K=K,P=P,Jb=Jb,
                nfact.V=nfact.V,
                nfact.W1=nfact.W1,
                nfact.W2=nfact.W2
                )

max.nfact.V <- 1+2*dim.V-sqrt(1+8*dim.V)
max.nfact.W1 <- 1+2*dim.W1-sqrt(1+8*dim.W1)
max.nfact.W2 <- 1+2*dim.W2-sqrt(1+8*dim.W2)

if (nfact.V > max.nfact.V) stop("Too many factors for V")
if (nfact.W1 > max.nfact.W1) stop("Too many factors for W1")
if (nfact.W2 > max.nfact.W2) stop("Too many factors for W2")
if (flags$W1.LKJ & nfact.W1>0) stop("Using LKJ prior on W1.  Set nfact.W1 to 0")

## priors

## We have to include these prior parameters

M20 <- matrix(0,1+P+Jb,J)
M20[1,] <- 10
M20[2:(Jb+1),] <- 0
M20[(Jb+2):(1+P+Jb),] <- 0
for (j in 1:J) {
    M20[Jb+1+j,j] <- -2
}
for (j in 1:Jb) {
    M20[j+1,j] <-  0
}

##C20 <- 1000*diag(1+P+Jb,1+P+Jb)
C20 <- diag(c(100,rep(1,Jb),rep(10,P)))

E.Sigma <-  diag(J) ## expected covariance across brands
nu0 <- P + 2*J + 5  ## must be greater than theta2 rows+cols
Omega0 <- (nu0-J-1)*E.Sigma

## The following priors are optional

if (flags$add.prior) {
    if (flags$full.phi) {

        ## prior on phi:  matrix normal with sparse covariances
        mean.phi <- matrix(0,Jb,J)
        cov.row.phi <- diag(Jb)
        cov.col.phi <- diag(J)
        chol.cov.row.phi <- t(chol(cov.row.phi))
        chol.cov.col.phi <- t(chol(cov.col.phi))

        prior.phi <- list(mean=mean.phi,
                          chol.row = chol.cov.row.phi,
                          chol.col = chol.cov.col.phi
                          )
    } else { ## diagonal phi
    ##    mean.phi <- rep(0.2,Jb);
    ##    sd.phi <- rep(0.1,Jb);

        prior.phi <- list(mean.mean=0,
                          sd.mean=0.5,
                          mode.var=1,
                          scale.var=1)

    } ## end diagonal phi




    if (flags$include.X) {
        ## prior on theta12:  matrix normal with sparse covariances
        mean.theta12 <- matrix(0,K,J)
        cov.row.theta12 <- 100*diag(K) ## across covariates within brand
        cov.col.theta12 <- 100*diag(J) ## across brand within covariates
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
    prior.delta <- list(a=1, b=3)

    ## For V, W1 and W2:   normal or truncated normal priors (if needed)
    if (!flags$fix.V) {
        prior.V <- list(diag.scale=1, diag.mode=1,
                         fact.scale=1, fact.mode=0)
    } else {
        prior.V <-  NULL
    }

    if (!flags$fix.W) {
        prior.W2 <- list(diag.scale=0.5, diag.mode=0,
                         fact.scale=1, fact.mode=0)
        if (flags$W1.LKJ) {
            ## LKJ prior on W1.
            ## Scale parameter has truncated(0) normal prior
            prior.W1 <- list(scale.mode=0, scale.s=0.5, eta=1)
        } else {
            prior.W1 <- list(diag.scale=0.5, diag.mode=0,
                             fact.scale=1, fact.mode=0)
        }
    } else {
        prior.W1 <- NULL
        prior.W2 <- NULL
    }

} else {
    prior.phi <- NULL
    prior.V <- NULL
    prior.W1 <- NULL
    prior.W2 <- NULL
    prior.delta <- NULL
    prior.theta12 <- NULL
}

tmp <- list(M20=M20,
            C20=C20,
            Omega0=Omega0,
            nu0=nu0,
            phi=prior.phi,
            theta12=prior.theta12,
            delta=prior.delta,
            V=prior.V,
            W1=prior.W1,
            W2=prior.W2
            )

priors <- Filter(function(x) !is.null(x), tmp)


## starting parameters

if (flags$include.X) {
    theta12.start <- matrix(0,K,J)
} else {
    theta12.start <- NULL
}

logit.delta.start <- 0

if (flags$full.phi) {
    phi.start <- matrix(0,Jb,J)
} else {
##    phi.start <- rep(0,Jb)
    phi.start <- rep(0,Jb+2)
}


if (flags$fix.V | flags$fix.W) {
    fixed.cov <- list(V=NULL, W=NULL)
} else {
    fixed.cov <- NULL
}

if (flags$fix.V) {
    fixed.cov$V <- 0.1*diag(N);
    V.start <- NULL
} else {
    V.length <- N + N*nfact.V - nfact.V*(nfact.V-1)/2
    V.start <- rep(0,V.length)
}

if (flags$fix.W) {
    fixed.cov$W <- .001*diag(1+J+P)
    W1.start <- NULL
    W2.start <- NULL
} else {
    if (flags$W1.LKJ) {
        W1.length <- dim.W1*(dim.W1-1)/2 + 1
    } else {
        W1.length <- dim.W1 + dim.W1*nfact.W1 - nfact.W1*(nfact.W1-1)/2
    }
    W1.start <- rep(0,W1.length)

    W2.length <- dim.W2 + dim.W2*nfact.W2 - nfact.W2*(nfact.W2-1)/2
    W2.start <- rep(0,W2.length)
}


tmp <- list(
    theta12=theta12.start,
    phi=phi.start,
    logit.delta=logit.delta.start,
    V=V.start,
    W1=W1.start,
    W2=W2.start
    )

start.list <- as.relistable(Filter(function(x) !is.null(x), tmp))

start <- unlist(start.list)


DL <- list(data=data, priors=priors,
           dimensions=dimensions,
           flags=flags,
           fixed.cov=fixed.cov)


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



## Need to bound variables to avoid overflow

opt1 <- optim(start,
              fn=get.f,
              gr=get.df,
              hessian=FALSE,
              method="BFGS",
              ##        lower = start-5,
              ##        upper=start+5,
              control=list(
                  fnscale=-1,
                  REPORT=1,
                  trace=3,
                  maxit=300
                  )
              )

opt2 <- trust.optim(opt1$par,
                    fn=get.f,
                    gr=get.df,
                    method="BFGS",
                    control=list(
                        report.level=5L,
                        report.precision=4L,
                        maxit=3000L,
                        function.scale.factor=-1,
                        preconditioner=1,
                        start.trust.radius=.01,
                        stop.trust.radius=1e-15,
                        contract.factor=.4,
                        expand.factor=2,
                        expand.threshold.radius=.85,
                        report.freq = 10L
                        )
                   )


opt <- trust.optim(opt2$solution,
                    fn=get.f,
                    gr=get.df,
                    method="BFGS",
                    control=list(
                        report.level=5L,
                        report.precision=4L,
                        maxit=3000L,
                        function.scale.factor=-1,
                        preconditioner=1,
                        start.trust.radius=.01,
                        stop.trust.radius=1e-1,
                        contract.factor=.4,
                        expand.factor=2,
                        expand.threshold.radius=.85,
                        report.freq = 10L
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

recover.W1.LKJ <- function(v, d) {
    W1_scale <- exp(v[1])
    p <- v[2:length(v)]
    stopifnot(length(p)==choose(d,2))
    chol_W1 <- CppADutils::lkj_unwrap_R(p, d)$L
    W1 <- tcrossprod(chol_W1)
    res <- W1_scale * W1
    return(res)
}

sol <- sol.vec
sol$delta <- exp(sol$logit.delta)/(1+exp(sol$logit.delta))

if (!flags$fix.V) {
    sol$V <- recover.cov.mat(sol.vec$V, dim.V, nfact.V)
}

if (!flags$fix.W) {
    sol$W2 <- recover.cov.mat(sol.vec$W2,dim.W2,nfact.W2)
    if (flags$W1.LKJ) {
        sol$W1 <- recover.W1.LKJ(sol.vec$W1,dim.W1)
    } else {
        sol$W1 <- recover.cov.mat(sol.vec$W1,dim.W1,nfact.W1)
    }
}

parcheck <- cl$par.check(opt$par)

cat("Computing Hessian\n")
hs <- get.hessian(opt$par)
cat("inverting negative Hessian\n")
cv <- solve(-hs)
se <- sqrt(diag(cv))
se.sol <- relist(se,skeleton=start.list)

##save(sol, se.sol, opt, dimensions, priors, flags, file=save.file)

