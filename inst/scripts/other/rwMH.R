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
scale <- 0.003
ndraws <- 10000
draws <- matrix(NA,ndraws,nvars)
logpost <- rep(NA,ndraws)
acc <- logical(ndraws)
track <- matrix(NA,ndraws,4)
colnames(track) <- c("f.curr","f.prop","log.r","log.u")
report <- 50

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
    if (i %% report == 0) cat("iter ",i,"\t");
    prop <- rMVN(1, x, prChol, FALSE)
    log.f2 <- get.f(prop)
    log.r <- min(log.f2-log.f1, 0)
    log.u <- log(runif(1))
    track[i,] <- c(log.f1, log.f2,log.r, log.u)

    if (log.u <= log.r) { ## accept
        x <- prop
        log.f1 <- log.f2
        if (i %% report == 0)   cat("ACCEPT\n")
        acc[i] <- TRUE
    } else {
        if (i %% report == 0)    cat("REJECT\n")
        acc[i] <- FALSE
    }
    draws[i,] <- x
    logpost[i] <- log.f1
}


