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


data.name <- "dpp"
data.is.sim <- FALSE

dn <- paste0("mcmod",data.name) ## name of data file, e.g., mcmoddpp
data(list=dn)  ## load data
mcmod <- eval(parse(text=dn)) ## rename to mcmod
save.file <- paste0("./nobuild/results/mode_",data.name,".Rdata")

if (data.is.sim) {
    flags <- mcmod$trueflags
} else {
    flags <- list(full.phi=FALSE, # default is a diagonal phi matrix
                  phi.re=FALSE,
                  add.prior=TRUE,
                  include.X=TRUE,
                  standardize=FALSE,
                  A.scale = 1,
                  fix.V1 = FALSE,
                  fix.V2 = FALSE,
                  fix.W = FALSE,
                  W1.LKJ = FALSE,
                  use.cr.pars = FALSE
                  )
}

nfact.V1 <- 0
nfact.V2 <- 0
nfact.W1 <- 0
nfact.W2 <- 0

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

##R <- mcmod$dimensions$R
R <- 1
if (flags$use.cr.pars) {
    CM <- mcmod$CM[1:T]
    ##CM <- mcmod$Ef[1:T]
} else {
    CM <- list()
##    CMcol <- 2
##    CM <- llply(mcmod$CM[1:T], function(x) return(x[,CMcol,drop=FALSE]))
    for(t in 1:T) CM[[t]] <- mcmod$Ef[[t]] + mcmod$Efl1[[t]]
}



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
             A=A, CM=CM) # E=E)

dim.V1 <- N
dim.V2 <- N*(1+P)
dim.W1 <- Jb+1
dim.W2 <- P
dim.W <- dim.W1+dim.W2

dimensions <- c(N=N,T=T,J=J,K=K,P=P,
                Jb=Jb,R=R,
                nfact.V1=nfact.V1,
                nfact.V2=nfact.V2,
                nfact.W1=nfact.W1,
                nfact.W2=nfact.W2
                )

max.nfact.V1 <- 1+2*dim.V1-sqrt(1+8*dim.V1)
max.nfact.V2 <- 1+2*dim.V2-sqrt(1+8*dim.V2)
max.nfact.W1 <- 1+2*dim.W1-sqrt(1+8*dim.W1)
max.nfact.W2 <- 1+2*dim.W2-sqrt(1+8*dim.W2)

if (nfact.V1 > max.nfact.V1) stop("Too many factors for V1")
if (nfact.V2 > max.nfact.V2) stop("Too many factors for V2")
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
C20 <- diag(c(1000,rep(1,Jb),rep(10,P)))
diag(C20)[1+1:Jb] <- 1

E.Sigma <-  diag(J) ## expected covariance across brands
nu0 <- P + 2*J + 6  ## must be greater than theta2 rows+cols
Omega0 <- (nu0-J-1)*E.Sigma

## The following priors are optional

if (flags$add.prior) {
    if (flags$full.phi) {

        ## prior on phi:  matrix normal with sparse covariances
        mean.phi <- matrix(0,Jb,J)
        cov.row.phi <- 1000*diag(Jb)
        cov.col.phi <- 1000*diag(J)
        chol.cov.row.phi <- t(chol(cov.row.phi))
        chol.cov.col.phi <- t(chol(cov.col.phi))

        prior.phi <- list(mean=mean.phi,
                          chol.row = chol.cov.row.phi,
                          chol.col = chol.cov.col.phi
                          )
    } else { ## diagonal phi
        if (flags$phi.re) {
            prior.phi <- list(mean.mean=0,
                              sd.mean=10,
                              mode.var=0,
                              scale.var=0.01)
        } else {
            mean.phi <- matrix(0,Jb,1)
            cov.col.phi <- diag(J)
            chol.cov.col.phi <- t(chol(cov.col.phi))
            prior.phi <- list(mean=mean.phi,
                              chol.col = chol.cov.col.phi
                              )
        }
    } ## end diagonal phi

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
    prior.delta <- list(a=1, b=3)

    ## For V1, V2, W1 and W2:   normal or truncated normal priors (if needed)
    if (!flags$fix.V1) {
        prior.V1 <- list(diag.scale=1, diag.mode=0,
                         fact.scale=0.1, fact.mode=0)
    } else {
        prior.V1 <-  NULL
    }

    if (!flags$fix.V2) {
        prior.V2 <- list(diag.scale=1, diag.mode=0,
                         fact.scale=1, fact.mode=0)
    } else {
        prior.V2 <-  NULL
    }

    if (!flags$fix.W) {
        prior.W2 <- list(diag.scale=1, diag.mode=0,
                         fact.scale=1, fact.mode=0)
        if (flags$W1.LKJ) {
            ## LKJ prior on W1.
            ## Scale parameter has truncated(0) normal prior
            prior.W1 <- list(scale.mode=0, scale.s=1, eta=1)
        } else {
            prior.W1 <- list(diag.scale=1, diag.mode=0,
                             fact.scale=1, fact.mode=0)
        }
    } else {
        prior.W1 <- NULL
        prior.W2 <- NULL
    }

    if (flags$use.cr.pars) {
        prior.creatives <- list(mean=rep(0,R-1),
                                chol.cov=t(chol(diag(R-1)))
                                )
    } else {
        prior.creatives <- NULL
    }

} else {
    prior.phi <- NULL
    prior.V1 <- NULL
    prior.V2 <- NULL
    prior.W1 <- NULL
    prior.W2 <- NULL
    prior.delta <- NULL
    prior.theta12 <- NULL
    prior.creatives <- NULL
}

tmp <- list(M20=M20,
            C20=C20,
            Omega0=Omega0,
            nu0=nu0,
            phi=prior.phi,
            theta12=prior.theta12,
            delta=prior.delta,
            V1=prior.V1,
            V2=prior.V2,
            W1=prior.W1,
            W2=prior.W2,
            creatives=prior.creatives
            )

priors <- Filter(function(x) !is.null(x), tmp)


## starting parameters

if (flags$include.X) {
    theta12.start <- matrix(0,K,J)
    ## theta12.start <- matrix(rnorm(K*J),K,J)
} else {
    theta12.start <- NULL
}

logit.delta.start <- 0.2

if (flags$full.phi) {
    phi.start <- matrix(0,Jb,J)
    ##phi.start <- matrix(rnorm(Jb*J),Jb,J)
} else {
    if (flags$phi.re) {
        phi.start <- rep(0,Jb+2)
        ## phi.start <- rnorm(Jb+2)
    } else {
        phi.start <- rep(0,Jb)
        ## phi.start <- rnorm(Jb)
    }
}


if (flags$fix.V1 | flags$fix.V2 | flags$fix.W) {
    fixed.cov <- list(V1=mcmod$truevals$V$V1,
                      V2=mcmod$truevals$V$VV,
                      W=mcmod$truevals$V$W)
} else {
    fixed.cov <- NULL
}

if (flags$fix.V1) {
    fixed.cov$V1 <- 0.1*diag(N);
    V1.start <- NULL
} else {
    V1.length <- N + N*nfact.V1 - nfact.V1*(nfact.V1-1)/2
    V1.start <- rep(0,V1.length)
}


if (flags$fix.V2) {
    fixed.cov$V2 <- 0.1*diag(N*(1+P));
    V2.start <- NULL
} else {
    V2.length <- N*(1+P) + N*nfact.V2 - nfact.V2*(nfact.V2-1)/2
    V2.start <- rep(0,V2.length)
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

if (flags$use.cr.pars) {
    creatives.start <- rep(0,R-1)
} else {
    creatives.start <- NULL
}

tmp <- list(
    theta12=theta12.start,
    phi=phi.start,
    logit.delta=logit.delta.start,
    V1=V1.start,
    V2=V2.start,
    W1=W1.start,
    W2=W2.start,
    creatives=creatives.start
    )

start.list <- as.relistable(Filter(function(x) !is.null(x), tmp))

start <- unlist(start.list)
nvars <- length(start)


DL <- list(data=data, priors=priors,
           dimensions=dimensions,
           flags=flags,
           fixed.cov=fixed.cov)

cat("Setting up\n")
cl <- new("ads", DL)
cat("Recording tape\n")
cl$record.tape(start)
cat("Objective function - taped\n")
f <- get.f(start)
cat("f = ",f,"\n")


## Need to bound variables to avoid overflow

opt1 <- optim(start,
              fn=get.f,
              gr=get.df,
              hessian=FALSE,
              method="BFGS",
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
                    method="SR1",
                    control=list(
                        report.level=5L,
                        report.precision=4L,
                        maxit=2000L,
                        function.scale.factor=-1,
                        preconditioner=0,
                        start.trust.radius=.01,
                        stop.trust.radius=1e-18,
                        cg.tol=1e-8,
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
                        maxit=5000L,
                        function.scale.factor=-1,
                        preconditioner=1,
                        start.trust.radius=.5,
                        stop.trust.radius=1e-17,
                        cg.tol=1e-8,
                        contract.factor=.8,
                        expand.factor=1.5,
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

if (!flags$fix.V1) {
    sol$V1 <- recover.cov.mat(sol.vec$V1, dim.V1, nfact.V1)
}

if (!flags$fix.V2) {
    sol$V2 <- recover.cov.mat(sol.vec$V2, dim.V2, nfact.V2)
}

if (!flags$fix.W) {
    sol$W2 <- recover.cov.mat(sol.vec$W2,dim.W2,nfact.W2)
    if (flags$W1.LKJ) {
        sol$W1 <- recover.W1.LKJ(sol.vec$W1,dim.W1)
    } else {
        sol$W1 <- recover.cov.mat(sol.vec$W1,dim.W1,nfact.W1)
    }
}

##parcheck <- cl$par.check(opt$par)

cat("Computing gradient\n")
gr <- get.df(opt$par)
cat("Computing Hessian\n")
hs <- get.hessian(opt$par)
cat("inverting negative Hessian\n")
cv <- solve(-hs)
se <- sqrt(diag(cv))
se.sol <- relist(se,skeleton=start.list)

## standard errors of products of phi and a

if (flags$use.cr.pars) {
    ax <- 1
    px <- seq(J*K+1, J*K + Jb*J)
    crx <- nvars-R+ax
    rg <- c(px,crx)
    cvg <- cv[rg,rg]
    GD <- matrix(0,Jb*J,Jb*J+1)
    GD[,1:(Jb*J)] <- diag(rep(sol$creatives[ax],Jb*J))
    GD[,(Jb*J+1)] <- as.vector(sol$phi)
    pa <- kronecker(as.vector(sol$phi), sol$creatives)
    cv.pa <- GD %*% cvg %*% t(GD)
    se.pa <- sqrt(diag(cv.pa))
}


## Standard errors of phi for real effects model

## if (flags$phi.re) {
##     px <- seq(J*K+1, J*K+Jb+2)
##     cv.phi <- cv[px,px]
##     GD <- matrix(0,Jb+2,Jb+2)
##     diag(GD)[1:Jb] <- exp(sol$phi[Jb+2]/2)
##     GD[Jb+1,Jb+1] <- 1
##     GD[Jb+2,Jb+2] <- exp(sol$phi[Jb+2]/2)/2
##     GD[1:Jb,Jb+1] <- exp(sol$phi[Jb+2]/2)*sol$phi[1:Jb]/2
##     GD[1:Jb,Jb+2] <- 1
##     s <- exp(sol$phi[Jb+2]/2)
##     pp <- s * sol$phi[1:3] + sol$phi[Jb+1]
##     sep <- GD %*% cv.phi %*% t(GD)
## }



save(sol, se.sol, opt, DL, gr, hs, data,file=save.file)

