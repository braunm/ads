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
if(run.par) registerDoParallel(cores=18) else registerDoParallel(cores=1)
seed.id <- 123
set.seed(seed.id)


data.name <- "ptw"
mode.file <- paste0("./nobuild/results/mode_test_",data.name,".Rdata")
save.file <- paste0("./nobuild/results/gds_test_",data.name,".Rdata")
##mode.file <- paste0("./nobuild/results/mode_",data.name,".Rdata")
##save.file <- paste0(".nobuild/results/gds_",data.name,".Rdata")


##----load post.mode, DL, gr, hs
load(mode.file)
scale <- 1.1
M <- 25000  ## proposal draws
n.draws <- 36  ## total number of draws needed
max.tries <- 100000  ## to keep sample.GDS from running forever
n.batch <- 12


##----defPropFuncs

rprop.wrap <- function(n.draws, params) {
    rMVN(n.draws, params$mean, params$CH, TRUE)
}
dprop.wrap <- function(d, params) {
    dMVN(matrix(d,ncol=length(params$mean)), params$mean, params$CH, TRUE)
}

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


##----propParams
cv <- solve(-hs)/scale^2
##CH <- Cholesky(-scale*as(forceSymmetric(hs),"sparseMatrix"))
CH <- t(chol(-scale*hs))
prop.params <- list(mean = post.mode, CH=CH)

##----proposals

cat("Sampling proposals\n")
log.c2 <- dprop.wrap(post.mode, prop.params)
draws.m <- as(rprop.wrap(M,prop.params),"matrix")
log.prop.m <- dprop.wrap(draws.m, params=prop.params)
cat("Computing posteriors for proposals\n")
log.post.m <- plyr::aaply(draws.m, 1, get.f, .parallel=run.par)

log.phi <- log.post.m - log.prop.m + log.c2 - log.c1
valid.scale <- all(log.phi <= 0)
stopifnot(valid.scale)


## ##----additional_proposals_for_testing
## cat("Additional proposals for testing\n")
## draws.m2 <- as(rprop.wrap(10*M,prop.params),"matrix")
## log.prop.m2 <- dprop.wrap(draws.m2, params=prop.params)
## cat("Computing posteriors for proposals\n")
## log.post.m2 <- plyr::aaply(draws.m2, 1, get.f, .parallel=run.par)
## log.phi2 <- log.post.m2 - log.prop.m2 + log.c2 - log.c1


##----sampleGDS_parallel
if (run.par) {
    print("Parallel")

    batch.size <- n.draws/n.batch
    n.batch <- floor(n.draws / batch.size)
    draws.list <- foreach(i=1:n.batch, .inorder=FALSE) %dopar% sample.GDS(
        n.draws = batch.size,
        log.phi = log.phi,
        post.mode = post.mode,
        fn.dens.post = get.f,
        fn.dens.prop = dprop.wrap,
        fn.draw.prop = rprop.wrap,
        prop.params = prop.params,
        report.freq = 100,
        thread.id = i,
        announce=TRUE,
        seed=as.integer(seed.id*i))
    ## combine results from each batch
    draws <- Reduce(function(x,y) Map(rbind,x,y), draws.list)
}

##----strDraws
str(draws)


##----summary
quants <-  plyr::aaply(draws[["draws"]], 2,
                       quantile, probs=c(.025, .5, .975),
                       .parallel = run.par)

save(draws, quants, log.phi, DL, file=save.file)
