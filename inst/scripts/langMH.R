rm(list=ls())
gc()

library(Matrix)
library(Rcpp)
library(RcppEigen)
library(bayesGDS)
library(plyr)
library(dplyr)
library(reshape2)
library(sparseMVN)
library(ggplot2)
library(stringr)

theme_set(theme_bw())

##----parallelSetup
library(doParallel, quietly=TRUE)
run.par <- TRUE
if(run.par) registerDoParallel(cores=3) else registerDoParallel(cores=1)


get.f <- function(P, ...) return(cl$get.f(P))
get.df <- function(P, ...) return(cl$get.fdf(P)$grad)
get.hessian <- function(P, ...) return(cl$get.hessian(P))
get.f.direct <- function(P, ...) return(cl$get.f.direct(P))
get.LL <- function(P, ...) return(cl$get.f.direct(P))
get.hyperprior <- function(P, ...) return(cl$get.hyperprior(P))

data.name <- "dpp"
mode.file <- paste0("./nobuild/results/mode_",data.name,".Rdata")
save.file <- paste0("./nobuild/results/langMH_",data.name,".Rdata")


##----load post.mode, DL, gr, hs
load(mode.file)
post.mode <- opt$par
cl <- new("ads", DL)
cl$record.tape(post.mode)
log.c1 <- get.f(post.mode)

nvars <- length(opt$par)
n.iter <- 1000
n.thin <- 10
n.draws <- floor(n.iter/n.thin)
restart <- TRUE
report <- 50
save.freq <- 2000 ## save if (i %% save.freq) == 0
sig <- 0.015

if (restart) {
    cat("restarting\n")
    load(save.file) ## loads x
    start.d <- NROW(draws)
    start.i <- tail(iter_draw,1) + 1
    end.i <- start.i + n.iter - 1
    draws <- rbind(draws,matrix(NA,n.draws,nvars))
    logpost <- c(logpost,rep(NA,n.draws))
    acc <- c(acc,logical(n.iter))
    track <- rbind(track,matrix(NA,n.iter,4))
    iter_draw <- c(iter_draw,rep(NA,n.draws))
} else {
    start.i <- 1
    start.d <- n.thin
    end.i <- n.iter
    x <- matrix(post.mode,nrow=1)
    draws <- matrix(NA,n.draws,nvars)
    logpost <- rep(NA,n.draws)
    acc <- logical(n.iter)
    track <- matrix(NA,n.iter,4)
    iter_draw <- rep(NA,n.draws)
}

log.fx <- get.f(as.vector(x))
grx <- get.df(as.vector(x))


colnames(track) <- c("f.curr","f.prop","log.r","log.u")



nm <- make_varnames(DL$dimensions)

rmvn.wrap <- function(n.draws, params) {
    rMVN(n.draws, params$mean, params$CH, TRUE)
}
dmvn.wrap <- function(d, params) {
    dMVN(matrix(d,ncol=length(params$mean)), params$mean, params$CH, TRUE)
}


prCov <- sig * solve(-hs)
prChol <- t(chol(prCov))

sampler_pars <- list(sig=sig, n.thin=n.thin, start.i=start.i,
                     start.d=start.d, n.draws=n.draws,
                     n.iter=n.iter)

d <- start.d
for (i in start.i:end.i) {
    if (i %% report == 0) cat("iter ",i,"\t");
    mx <- t(x) + 0.5 * prCov %*% grx
    y <- rMVN(1, mx, prChol, FALSE)
    log.py <- dMVN(y, mx, prChol, FALSE)
    log.fy <- get.f(y)

    gry <- get.df(y)
    my <- t(y) + 0.5 * prCov %*% gry
    log.px <- dMVN(x, my, prChol, FALSE)

    log.r <- min(log.fy-log.fx + log.px-log.py, 0)
    log.u <- log(runif(1))
    track[i,] <- c(log.fx, log.fy,log.r, log.u)

    if (log.u <= log.r) { ## accept
        x <- y
        log.fx <- log.fy
        grx <- gry
        if (i %% report == 0)   cat("ACCEPT\n")
        acc[i] <- TRUE
    } else {
        if (i %% report == 0)    cat("REJECT\n")
        acc[i] <- FALSE
    }

    if ((i %% n.thin)==0) {
        d <- d+1
        iter_draw[d] <- i
        draws[d,] <- x
        logpost[d] <- log.fx
    }

    if ((i %% save.freq)==0) { ## interim save
        dimnames(draws) <- list(iter=iter_draw, var=nm)
        save(draws, logpost, acc, track, sig, x,
             nm, iter_draw, sampler_pars,
             file=save.file)
    }
}


## save at end
dimnames(draws) <- list(iter=iter_draw, var=nm)
save(draws, logpost, acc, track, sig, x,
     nm, iter_draw, sampler_pars,
     file=save.file)


