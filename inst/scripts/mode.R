rm(list=ls())
gc()

library(Matrix)
library(Rcpp)
library(RcppEigen)
library(numDeriv)
library(trustOptim)
library(plyr)
library(reshape2)
library(Rcgmin)


set.seed(1234)

mod.name <- "hdlm"
data.name <- "sim"

##data.file <- paste0("~/Documents/hdlm/ads/data/mcmod",data.name,".RData")
## save.file <- paste0("~/Documents/hdlm/results/",mod.name,"_",data.name,"_mode.Rdata")

data.file <- paste0("data/mcmod",data.name,".RData")
save.file <- paste0("inst/results/",mod.name,"_",data.name,"_mode.Rdata")

flags <- list(include.phi=FALSE,
              include.c=TRUE,
              include.u=TRUE,
              add.prior=TRUE,
              include.X=TRUE,  
              A.scale = 1000000,
       ##       A.scale = 1,
              W1.LKJ = FALSE,  # W1.LKJ true means we do LKJ, otherwise same as W2
              fix.V = FALSE,
              fix.W = FALSE,
              estimate.M20 = TRUE,
              estimate.asymptote = TRUE
              )

nfact.V <- 0
nfact.W1 <- 0
nfact.W2 <- 0

get.f <- function(P, ...) return(cl$get.f(P))
get.df <- function(P, ...) return(cl$get.fdf(P)$grad)
get.hessian <- function(P, ...) return(cl$get.hessian(P))
get.f.direct <- function(P, ...) return(cl$get.f.direct(P))
get.LL <- function(P, ...) return(cl$get.f.direct(P))
get.hyperprior <- function(P, ...) return(cl$get.hyperprior(P))

load(data.file)
#print("hello")
#dn <- paste0("mcmod",data.name) 
#data(list=dn)
#mcmod <- eval(parse(text=dn))

N <- mcmod$dimensions$N
T <- mcmod$dimensions$T
J <- mcmod$dimensions$J

if (flags$include.X) K <- mcmod$dimensions$K else K <- 0
P <- mcmod$dimensions$P

Y <- mcmod$Y[1:T]


F1 <- mcmod$F1[1:T]
F2 <- mcmod$F2[1:T]
A <- mcmod$A[1:T]
##A <- llply(mcmod$A[1:T],function(x) return(x/flags$A.scale))
## A2 <- as.relistable(A2)
## tmp <- unlist(A2)
## mean.A <- mean(tmp[tmp>0])
## sd.A <- sd(tmp[tmp>0])
## tmp[tmp>0] <- (tmp[tmp>0]-mean.A)/sd.A
## tmp[tmp!=0] <- tmp[tmp!=0] - min(tmp[tmp!=0]) + 1
## A <- relist(tmp)

if (flags$include.phi) {
  E <- mcmod$E[1:T]
} else {
  E <- NULL
}

if (flags$include.X) {
  X <- mcmod$X[1:T]
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


dim.V <- N
dim.W1 <- J+1
dim.W2 <- P
dim.W <- dim.W1+dim.W2

dimensions <- c(N=N,T=T,J=J,K=K,P=P,
                nfact.V=nfact.V,
                nfact.W1=nfact.W1,
                nfact.W2=nfact.W2
                )

max.nfact.V <- 1+2*dim.V-sqrt(1+8*dim.V)
max.nfact.W1 <- 1+2*dim.W1-sqrt(1+8*dim.W1)
max.nfact.W2 <- 1+2*dim.W2-sqrt(1+8*dim.W2)

if (nfact.V > max.nfact.V) stop("Too many factors for V")
if (nfact.W2 > max.nfact.W2) stop("Too many factors for W1")
if (nfact.W2 > max.nfact.W2) stop("Too many factors for W2")


## priors

## We have to include these prior parameters

M20 <- matrix(0,1+P+J,J)
M20[1,] <- 15
M20[2:(J+1),] <- -.005
M20[(J+2):(1+P+J),] <- 1
for (j in 1:J) {
    M20[J+1+j,j] <- -2
    M20[j+1,j] <- .25
}

if (flags$estimate.M20) {
    M20.mean <- M20
    M20.cov.row <- 1000*diag(1+P+J)
    M20.cov.col <- 1000*diag(J)
    prior.M20 <- list(mean=M20,
                      chol.row = t(chol(M20.cov.row)),
                      chol.col = t(chol(M20.cov.col))
                      )
    
} else {
    prior.M20 <- list(M20=M20)
}


if (flags$estimate.asymptote) {
    asymp.cov.row <- 500*diag(J)
    asymp.cov.col <- 500*diag(J)
    prior.asymp <- list(
                      chol.row = t(chol(asymp.cov.row)),
                      chol.col = t(chol(asymp.cov.col))
                      )
    
} else {
    prior.asymp <- list(asymp=M20[2:(J+1),])
}

C20 <- 50*diag(1+P+J,1+P+J)


E.Sigma <- 0.1 * diag(J) ## expected covariance across brands
nu0 <- P + 2*J + 6  ## must be greater than theta2 rows+cols
Omega0 <- (nu0-J-1)*E.Sigma

## The following priors are optional

if (flags$add.prior) { 
    if (flags$include.phi) {
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
    } ## end include.phi
    
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
    
    if (flags$include.c) {
        prior.log.c.mean <- -.5       
        prior.log.c <- list(mean.mean=prior.log.c.mean,
                        mean.sd=5,
                        sd.mean=5,
                        sd.sd=5)
    } else {
        prior.log.c <- NULL;
    }

    if (flags$include.u) {
        prior.log.u.mean <- -3       
        prior.log.u <- list(mean.mean=prior.log.u.mean,
                            mean.sd=1,
                            sd.mean=5,
                            sd.sd=1)
    } else {
        prior.log.u <- NULL;
    }

    
    ## For V, W1 and W2:   normal or truncated normal priors (if needed)

    if (!flags$fix.V) {        
        prior.V <- list(diag.scale=1, diag.mode=1,
                         fact.scale=1, fact.mode=0)
    } else {
        prior.V <-  NULL
    }

    if (!flags$fix.W) {
        prior.W2 <- list(diag.scale=1, diag.mode=1,
                         fact.scale=.01, fact.mode=0)
        
        ## LKJ prior on W1.  Scale parameter has truncated(0) normal prior
        
        if (flags$W1.LKJ) {
            prior.W1 <- list(scale.mode=0, scale.s=1, eta=1)
        } else {
            prior.W1 <- list(diag.scale=.01, diag.mode=0,
                             fact.scale=.01, fact.mode=0)
        }
    } else {
        prior.W1 <- NULL
        prior.W2 <- NULL
    }
    
} else {    
    prior.M20 <- NULL
    prior.asymp <- NULL
    prior.phi <- NULL
    prior.V <- NULL
    prior.W1 <- NULL
    prior.W2 <- NULL
    prior.delta <- NULL
    prior.log.c <- NULL
    prior.log.u <- NULL
    prior.theta12 <- NULL
}

tmp <- list(M20=prior.M20,
            asymp = prior.asymp,
            C20=C20,
            Omega0=Omega0,
            nu0=nu0,
            phi=prior.phi,
            theta12=prior.theta12,
            delta=prior.delta,
            log.c=prior.log.c,
            log.u=prior.log.u,
            V=prior.V,
            W1=prior.W1, W2=prior.W2
            )

priors <- Filter(function(x) !is.null(x), tmp)


## starting parameters


if (flags$estimate.M20) {
    M20.start <- M20
} else {
    M20.start <- NULL
}

if (flags$estimate.asymp) {
    asymp.start <- M20[2:(J+1),]
} else {
    asymp.start <- NULL
}

if (flags$include.X) {
    theta12.start <- matrix(0,K,J)   
} else {
    theta12.start <- NULL
}

if (flags$include.c) {
    log.c.mean.log.sd.start <- c(-.5, 1)
    log.c.off.start <- rep(0,J)
} else {
    log.c.mean.log.sd.start <- c.off.start <- NULL
    log.c.off.start <- NULL
}



if (flags$include.u) {
    log.u.mean.log.sd.start <- c(-.5, 1)
    log.u.off.start <- rep(0,J)
} else {
    log.u.mean.log.sd.start <- u.off.start <- NULL
    log.u.off.start <- NULL
}
 
logit.delta.start <- 0    



if (flags$include.phi) {
    phi.start <- matrix(0,J,J)
} else {
    phi.start <- NULL
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
    ##V.start <- (1:V.length)/10
    ## V.start <- rnorm(V.length) - 3  
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
        W1.length <- (J+1) + (J+1)*nfact.W1 - nfact.W1*(nfact.W1-1)/2
    }
    ## W1.start <- (1:W1.length)/20
    ## W1.start <- rnorm(W1.length) - 3
    W1.start <- rep(0,W1.length)

    W2.length <- P + P*nfact.W2 - nfact.W2*(nfact.W2-1)/2
    ## W2.start <- (2:(W2.length+1))/30
    ## W2.start <- rnorm(W2.length) - 3
    W2.start <- rep(0,W2.length)
}



tmp <- list(
    M20 = M20.start,
    asymp = asymp.start,
    theta12=theta12.start,
    log.c.mean.log.sd = log.c.mean.log.sd.start,
    log.c.off = log.c.off.start,
    log.u.mean.log.sd = log.u.mean.log.sd.start,
    log.u.off = log.u.off.start,
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


opt1 <- optim(start,
             fn=get.f,
             gr=get.df,
             hessian=FALSE,
             method="BFGS",
             control=list(
               fnscale=-1,
               REPORT=5,
               trace=3,
               maxit=300
               )
             )

opt2 <- opt1

## opt2 <- Rcgmin(opt1$par,
##                fn=function(x) -get.f(x),
##                gr=function(x) -get.df(x),
##                control=list(trace=2,
##                    maxit=31)
##                )

opt3 <- trust.optim(opt2$par,                    
                    fn=get.f,
                    gr=get.df,       
                    method="BFGS",
                    control=list(
                        report.level=5L,
                        report.precision=4L,
                        maxit=3000L,
                        function.scale.factor=-1,
                        preconditioner=0,
                        stop.trust.radius=1e-12,
                        contract.factor=.4,
                        expand.factor=2,
                        expand.threshold.radius=.85,
                        report.freq = 10L
                        )
                    )


opt <- trust.optim(opt3$solution,
                   fn=get.f,
                   gr=get.df,
                   method="SR1",
                   control=list(
                       report.level=5L,
                       report.precision=4L,
                       maxit=3000L,
                       function.scale.factor=-1,
                       preconditioner=0,
                       stop.trust.radius=1e-12,
                       contract.factor=.7,
                       report.freq=10L
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

if (!flags$fix.V) {
    sol$V <- recover.cov.mat(sol.vec$V, dim.V, nfact.V)
}

if (!flags$fix.W) {
    sol$W2 <- recover.cov.mat(sol.vec$W2,dim.W2,nfact.W2)
    if(flags$W1.LKJ) {
        sol$W1 <- recover.corr.mat(sol.vec$W1,dim.W1)
    } else {
        sol$W1 <- recover.cov.mat(sol.vec$W1,dim.W1,nfact.W1)
    }
}

if (flags$include.c){
        sol$cj <- exp(exp(sol$log.c.mean.log.sd[2]) * sol$log.c.off + sol$log.c.mean.log.sd[1])
 }

if (flags$include.u){
            sol$uj <- exp(exp(sol$log.u.mean.log.sd[2]) * sol$log.u.off + sol$log.u.mean.log.sd[1])
}



parcheck <- cl$par.check(opt$par)

cat("Computing Hessian\n")
hs <- get.hessian(opt$par)
cat("inverting negative Hessian\n")
cv <- solve(-hs)
se <- sqrt(diag(cv))
se.sol <- relist(se,skeleton=start.list)

save(sol, se.sol, opt, dimensions, priors, flags, file=save.file)
