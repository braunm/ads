rm(list=ls())
gc()

library(Matrix)
library(Rcpp)
library(RcppEigen)
library(bayesGDS)
library(plyr)
library(reshape2)
library(sparseMVN)

##----parallelSetup
library(doParallel, quietly=TRUE)
run.par <- TRUE
if(run.par) registerDoParallel(cores=3) else registerDoParallel(cores=1)
seed.id <- 123
set.seed(seed.id)

get.f <- function(P, ...) return(cl$get.f(P))
get.df <- function(P, ...) return(cl$get.fdf(P)$grad)
get.hessian <- function(P, ...) return(cl$get.hessian(P))
get.f.direct <- function(P, ...) return(cl$get.f.direct(P))
get.LL <- function(P, ...) return(cl$get.f.direct(P))
get.hyperprior <- function(P, ...) return(cl$get.hyperprior(P))

data.name <- "dpp"
mode.file <- paste0("./nobuild/results/mode_",data.name,".Rdata")
save.file <- paste0(".nobuild/results/gds_",data.name,".Rdata")


##----load post.mode, DL, gr, hs
load(mode.file)
post.mode <- opt$par
cl <- new("ads", DL)
cl$record.tape(post.mode)
log.c1 <- get.f(post.mode)

nvars <- length(opt$par)
scale <- 1
ndraws <- 100
draws <- matrix(NA,ndraws,nvars)

rmvn.wrap <- function(n.draws, params) {
    rMVN(n.draws, params$mean, params$CH, TRUE)
}
dmvn.wrap <- function(d, params) {
    dMVN(matrix(d,ncol=length(params$mean)), params$mean, params$CH, TRUE)
}

prMean <- post.mode
prCov <- scale*solve(-hs)
prChol <- t(chol(prCov))

x <- post.mode
log.f1 <- get.f(x)

for (i in 1:ndraws) {
    cat("iter ",i,"\t");
    prop <- rMVN(1, post.mode, prChol, TRUE)
    log.f2 <- get.f(prop)
    r <- min(log.f1-log.f2, 0)
    log.u <- log(runif(1))

    if (log.u <= r) { ## accept
        x <- prop
        log.f1 <- log.f2
        cat("ACCEPT\n")
    } else {
        cat("REJECT\n")
    }
    draws[i,] <- x
}


