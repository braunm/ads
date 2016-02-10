rm(list=ls())
gc()

library(Matrix)
library(Rcpp)
library(RcppEigen)
library(sparseMVN)
library(bayesGDS)
library(plyr)
library(reshape2)

##----parallelSetup
library(doParallel, quietly=TRUE)
run.par <- TRUE
if(run.par) registerDoParallel(cores=2) else registerDoParallel(cores=1)
seed.id <- 123
set.seed(seed.id)


data.name <- "dpp"
mode.file <- paste0("../../results/mode_",data.name,".Rdata")
save.file <- paste0("../../results/gds_",data.name,".Rdata")


##----load post.mode, DL, gr, hs
load(mode.file)
scale <- .97
M <- 50000  ## proposal draws
n.draws <- 4  ## total number of draws needed
max.tries <- 10000  ## to keep sample.GDS from running forever
n.batch <- 2


##----defPropFuncs


get.f <- function(P, ...) return(cl$get.f(P))
get.df <- function(P, ...) return(cl$get.fdf(P)$grad)
get.hessian <- function(P, ...) return(cl$get.hessian(P))
get.f.direct <- function(P, ...) return(cl$get.f.direct(P))
get.LL <- function(P, ...) return(cl$get.f.direct(P))
get.hyperprior <- function(P, ...) return(cl$get.hyperprior(P))

post.mode <- opt$par
cl <- new("ads", DL)
cl$record.tape(post.mode)
log.c1 <- get.f(post.mode)


rmvn.sparse.wrap <- function(n.draws, params) {
    rmvn.sparse(n.draws, params[["mean"]], params[["CH"]], prec=TRUE)
}
dmvn.sparse.wrap <- function(d, params) {
    dmvn.sparse(d, params[["mean"]], params[["CH"]], prec=TRUE)
}


##----propParams

chol.hess <- Cholesky(-scale*hs)
prop.params <- list(mean = post.mode, CH = chol.hess)

##----proposals

log.c2 <- dmvn.sparse.wrap(post.mode, prop.params)
draws.m <- as(rmvn.sparse.wrap(M,prop.params),"matrix")
log.post.m <- plyr::aaply(draws.m, 1, FD$fn, .parallel=run.par)
log.prop.m <- dmvn.sparse.wrap(draws.m, params=prop.params)
log.phi <- log.post.m - log.prop.m + log.c2 - log.c1
valid.scale <- all(log.phi <= 0)
stopifnot(valid.scale)
##----sampleGDS_parallel
if (run.par) {
    print("Parallel")

    batch.size <- n.draws/n.batch
    n.batch <- floor(n.draws / batch.size)
    draws.list <- foreach(i=1:n.batch, .inorder=FALSE) %dopar% sample.GDS(
        n.draws = n.draws,
        log.phi = log.phi,
        post.mode = post.mode,
        fn.dens.post = FD$fn,
        fn.dens.prop = dmvn.sparse.wrap,
        fn.draw.prop = rmvn.sparse.wrap,
        prop.params = prop.params,
        report.freq = 1000,
        thread.id = i,
        announce=TRUE,
        seed=as.integer(seed.id*i))
    ## combine results from each batch
    draws <- Reduce(function(x,y) Map(rbind,x,y), draws.list)
}

##----strDraws
str(draws)


##----summary
quants <-  plyr::aaply(draws[["draws"]][,(N*k+1):nvars], 2,
                       quantile, probs=c(.025, .5, .975),
                       .parallel = run.par)
