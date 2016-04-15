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

##----should we run the mode.R script? Otherwise pull results from another run
runmode <- FALSE

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

data.name <- "ptw"

if(runmode) source("./inst/scripts/mode.R")
mode.file <- paste0("./nobuild/results/mode_",data.name,".Rdata")

##----pre parallel constants

n.iter <- 500
n.thin <- 50
n.draws <- floor(n.iter/n.thin)
n.chains <- 3
restart <- TRUE
report <- 100
save.freq <- 1000 ## save if (i %% save.freq) == 0
sig <- 0.015


#----needed functions (could be moved?)
rmvn.wrap <- function(n.draws, params) {
    rMVN(n.draws, params$mean, params$CH, TRUE)
}
dmvn.wrap <- function(d, params) {
    dMVN(matrix(d,ncol=length(params$mean)), params$mean, params$CH, TRUE)
}



sample.MH <- function(DL, restart, mode.file, save.file, n.draws, report, n.iter, n.thin, save.freq, sig, seed, thread.id){
    set.seed(seed)
    save.file <- paste0("./nobuild/results/langMH_",data.name,"_", thread.id,".Rdata")
    load(mode.file)
    post.mode <- opt$par
    cl <- new("ads", DL)
    cl$record.tape(post.mode)
    log.c1 <- cl$get.f(post.mode)
    
    nvars <- length(opt$par)
    
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
        start.d <- 0
        end.i <- n.iter
        x <- matrix(post.mode,nrow=1)
        draws <- matrix(NA,n.draws,nvars)
        logpost <- rep(NA,n.draws)
        acc <- logical(n.iter)
        track <- matrix(NA,n.iter,4)
        iter_draw <- rep(NA,n.draws)
    }
    
    log.fx <- cl$get.f(as.vector(x))
    grx <- cl$get.df(as.vector(x))
    
    colnames(track) <- c("f.curr","f.prop","log.r","log.u")
    
    nm <- make_varnames(DL$dimensions)
    
    prCov <- sig * solve(-hs)
    prChol <- t(chol(prCov))
    
    sampler_pars <- list(sig=sig, n.thin=n.thin, start.i=start.i,
    start.d=start.d, n.draws=n.draws,
    n.iter=n.iter)
    
    d <- start.d
    for (i in start.i:end.i) {
        if (i %% report == 0 & thread.id == 1) cat("iter :",i,"\n================\n");
        mx <- t(x) + 0.5 * prCov %*% grx
        y <- rMVN(1, mx, prChol, FALSE)
        log.py <- dMVN(y, mx, prChol, FALSE)
        log.fy <- cl$get.f(y)
        
        gry <- cl$get.df(y)
        my <- t(y) + 0.5 * prCov %*% gry
        log.px <- dMVN(x, my, prChol, FALSE)
        
        log.r <- min(log.fy-log.fx + log.px-log.py, 0)
        log.u <- log(runif(1))
        track[i,] <- c(log.fx, log.fy,log.r, log.u)
        
        if (log.u <= log.r) { ## accept
            x <- y
            log.fx <- log.fy
            grx <- gry
            if (i %% report == 0)   cat(paste0("ACCEPT, chain: ", thread.id,  "\n"))
            acc[i] <- TRUE
        } else {
            if (i %% report == 0)    cat(paste0("REJECT, chain: ", thread.id,"\n"))
            acc[i] <- FALSE
        }

## browser()

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

    return(list(draws=draws, logpost=logpost, track=track))
}   ## end of sampling for a single chain

# j <- 1

# test
##sample.MH(DL = DL, restart = restart, mode.file = mode.file, save.file = save.file, n.draws = n.draws, report = report, n.iter = n.iter, n.thin = n.thin, save.freq = save.freq, sig = sig, thread.id = j)


##----run parallel chains
##----load post.mode, DL, gr, hs
if(run.par) {
    print("Parallel")
    
    draws.list <- foreach(j = 1:n.chains, .inorder = FALSE) %dopar% sample.MH(
        DL = DL,
        restart = restart,
        mode.file = mode.file,
        save.file = save.file,
        n.draws = n.draws,
        report = report,
        n.iter = n.iter,
        n.thin = n.thin,
        save.freq = save.freq,
        sig = sig,
        seed = j,
        thread.id = j
    )
}

## separate elements in the list
draws <- list()
logpost <- NULL
for(i in 1:n.chains) {
        draws[[i]] <- draws.list[[i]]$draws
        logpost <- cbind(logpost, draws.list[[i]]$logpost)
}

## save final output
save(draws, logpost, mcmod, file = paste0("./nobuild/results/langMH_",data.name,".RData"))




