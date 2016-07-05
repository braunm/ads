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
##library(doParallel, quietly=TRUE)
##run.par <- TRUE
##if(run.par) registerDoParallel(cores=4) else registerDoParallel(cores=1)

get.f <- function(P, ...) return(cl$get.f(P))
get.df <- function(P, ...) return(cl$get.fdf(P)$grad)
get.hessian <- function(P, ...) return(cl$get.hessian(P))
get.f.direct <- function(P, ...) return(cl$get.f.direct(P))
get.LL <- function(P, ...) return(cl$get.f.direct(P))
get.hyperprior <- function(P, ...) return(cl$get.hyperprior(P))

data.name <- "dpp"

if(runmode) source("./inst/scripts/mode.R")
mode.file <- paste0("./nobuild/results/mode_test_",data.name,".Rdata")

##----pre parallel constants

n.iter <- 20
n.thin <- 1
n.draws <- floor(n.iter/n.thin)
n.chains <- 1
restart <- FALSE         ## if true, it will continue where the process left off
report <- 1
save.freq <- 1 ## save if (i %% save.freq) == 0
sig <- 0.015


thread.id <- 1
save.file <- paste0("./nobuild/results/mcmc_test",data.name,"_", thread.id,".Rdata")
load(mode.file)
post.mode <- opt$par
nvars <- length(post.mode)
nm <- make_varnames(DL$dimensions, DL$flags)

## MH proposal draws
prCov <- sig * solve(-hs)
prChol <- t(chol(prCov))


ch <- ads::chain$new(post.mode, DL=DL,id=thread.id,
                seed=123, hs.mode=hs, sig=sig)

cat("iterating\n")
ch$iterate(10)  ## first 10 draws
ch$iterate(5) ## next 5 draws


stop()
if ((i %% save.freq)==0) { ## interim save
    dimnames(draws) <- list(iter=iter_draw, var=nm)
    save(draws, logpost, acc, track, sig, x,
         nm, iter_draw, sampler_pars,
         file=save.file)
}




    ## save at end
    dimnames(draws) <- list(iter=iter_draw, var=nm)
    save(draws, logpost, acc, track, sig, x,
    nm, iter_draw, sampler_pars,
    file=save.file)

    return(list(draws=draws, logpost=logpost, track=track))
   ## end of sampling for a single chain


