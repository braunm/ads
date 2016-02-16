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
save.file <- paste0("./nobuild/results/langMH_",data.name,".Rdata")




##----load post.mode, DL, gr, hs
load(mode.file)
post.mode <- opt$par
cl <- new("ads", DL)
cl$record.tape(post.mode)
log.c1 <- get.f(post.mode)

nvars <- length(opt$par)
ndraws <- 5000
restart <- TRUE

if (restart) {
    cat("restarting\n")
    load(save.file) ## loads x
    start.i <- NROW(draws)+1
    end.i <- start.i + ndraws - 1
    draws <- rbind(draws,matrix(NA,ndraws,nvars))
    logpost <- c(logpost,rep(NA,ndraws))
    acc <- c(acc,logical(ndraws))
    track <- rbind(track,matrix(NA,ndraws,4))
} else {
    start.i <- 1
    end.i <- ndraws
    x <- matrix(post.mode,nrow=1)
    draws <- matrix(NA,ndraws,nvars)
    logpost <- rep(NA,ndraws)
    acc <- logical(ndraws)
    track <- matrix(NA,ndraws,4)
}

log.fx <- get.f(as.vector(x))
grx <- get.df(as.vector(x))


colnames(track) <- c("f.curr","f.prop","log.r","log.u")
report <- 1

sig <- 0.025

rmvn.wrap <- function(n.draws, params) {
    rMVN(n.draws, params$mean, params$CH, TRUE)
}
dmvn.wrap <- function(d, params) {
    dMVN(matrix(d,ncol=length(params$mean)), params$mean, params$CH, TRUE)
}


prCov <- sig * solve(-hs)
prChol <- t(chol(prCov))


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
    draws[i,] <- x
    logpost[i] <- log.fx
}



nm <- make_varnames(DL$dimensions)
dimnames(draws) <- list(iter=1:NROW(draws),var=nm)


save(draws, logpost, acc, track, sig, x,
     nm, file=save.file)

F <- melt(draws) %>%
  tidyr::separate(var, into=c("par","D1","D2"), sep="\\.",
                  remove=FALSE, fill="right")






Fphi <- dplyr::filter(F,par=="phi" & iter>20000)
trace_phi <- ggplot(Fphi, aes(x=iter,y=value)) %>%
  + geom_line(size=.1) %>%
  + geom_hline(yintercept=0,color="red") %>%
  + facet_grid(D1~D2, scales="free")
print(trace_phi)

## Ftheta12 <- dplyr::filter(F,str_detect(var,"theta12"))
## trace_theta12 <- ggplot(Ftheta12, aes(x=iter,y=value)) %>%
##   + geom_line(size=.1) %>%
##   + facet_wrap(~var, scales="free")
## print(trace_theta12)


trace_logpost <- data_frame(iter=1:length(logpost),
                       value=logpost) %>%
  ggplot(aes(x=iter, y=value)) %>%
  + geom_line(size=.1)
print(trace_logpost)



