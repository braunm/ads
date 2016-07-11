#' A single langMH chain
#'
#' @docType class
#' @importFrom R6 R6Class
#' @export
#' @field DL data list
#' @field id identifier of the chain
#' @field seed a starting seed for the chain
#' @field nvars number of variables
#' @field acc_rate cumulative acceptance rate
#' @field x current state of the sampler (most recent draw)
#' @field log.fx log f(x)
#' @field grx gradient at x
#' @field draws all collected draws
#' @field acc vector of whether an iteration was accepted or not
#' @field iter iteration ids for each draw
#' @field logpost log posterior for each draw
#' @field last_iter iteration id of most recent draw
#' @field last_draw draw id of most recent draw
#' @field varnames vector of variable names for draws
#' @field e1,e2,A1,delta adaptation parameters
#' @field mu,G,sig,norm.grx internal adaptation parameters
#' @field opt_acc optimal acceptance rate
chain <- R6Class("chain",
                 public=list(
                     DL = NA,
                     id = 1,
                     seed = 1,
                     nvars = NA,
                     acc_rate = NA,
                     x = NA,
                     log.fx = NA,
                     grx = NA,
                     draws = NULL,
                     acc = NULL,
                     iter = NULL,
                     step = NULL,
                     logpost = NULL,
                     last_iter = 0,
                     last_draw = 0,
                     varnames = NA,
                     cl = NA,
                     log.r = NA,
                     mu = NA,
                     G = NA,
                     Lam = NA,
                     sig = NA,
                     delta = NA,
                     opt_acc = NA,
                     norm.grx = NA,
                     e1 = NA, e2 = NA,
                     A1 = NA,
                     scales = NA,
                     initialize = function(start, DL, id, seed, hs.mode,
                                           adaptList) {
                         "Initializing function"
                         self$DL <- DL
                         self$id <- id
                         self$seed <- seed
                         self$cl <- new("ads",DL)
                         self$cl$record.tape(start)
                         self$nvars <- length(start)
                         self$x <- matrix(start,nrow=1)
                         self$log.fx <- self$cl$get.f(start)
                         self$grx <- self$cl$get.df(start)
                         self$norm.grx <- sqrt(sum(self$grx^2))
                         self$draws <- NULL
                         self$logpost <- NULL
                         self$acc <- NULL
                         set.seed(self$seed)
                         self$sig <- adaptList$sig
                         self$mu <- start
                         self$G <- solve(-hs.mode)
                         self$e1 <- adaptList$e1
                         self$e2 <- adaptList$e2
                         self$A1 <- adaptList$A1
                         self$delta <- adaptList$delta
                         self$step <- adaptList$step
                         self$opt_acc <- adaptList$opt_acc
                         self$Lam <- self$G + self$e2*diag(self$nvars)
                         self$last_iter <- 0
                         self$last_draw <- 0
                         self$varnames <- make_varnames(DL$dimensions, DL$flags)
                     },
                     sample = function() {
                         Dx <- self$grx * self$delta / max(self$delta,self$norm.grx)
                         mx <- as.vector(self$x + 0.5 * self$sig^2 * Dx %*% self$Lam)
                         prCov <- self$sig^2 * self$Lam
                         prChol <- t(chol(prCov))
                         y <- rMVN(1, mx, prChol, FALSE) ## row matrix
                         log.py <- dMVN(y, mx, prChol, FALSE)
                         log.fy <- self$cl$get.f(as.vector(y))

                         gry <- self$cl$get.df(as.vector(y)) ## vector
                         norm.gry <- sqrt(sum(gry^2))
                         Dy <- gry * self$delta / max(self$delta,norm.gry)
                         my <- as.vector(t(y) + 0.5 * self$sig^2 * self$Lam %*% Dy)
                         log.px <- dMVN(self$x, my, prChol, FALSE)

                         self$log.r <- min(log.fy-self$log.fx + log.px-log.py, 0)
                         log.u <- log(runif(1))
                         if (log.u <= self$log.r) { ## accept
                             self$x <- y
                             self$log.fx <- log.fy
                             self$grx <- gry
                             self$norm.grx <- norm.gry
                             acc <- TRUE
                         } else {
                             acc <- FALSE
                         }
                         return(acc)
                     },
                     iterate = function(n.iter, n.thin=1,report_freq=1,...) {
                         L <- private$iterate_batch(n.iter, n.thin, report_freq,...)
                         self$last_iter <- self$last_iter + n.iter
                         self$last_draw <- self$last_draw + n.iter/n.thin
                         self$iter <- c(self$iter, L$iter)
                         self$draws <- rbind(self$draws,L$dr) ## not efficient
                         self$logpost <- c(self$logpost, L$logpost)
                         self$acc <- c(self$acc, L$acc)
                         self$acc_rate <- mean(self$acc)
                         self$scales <- c(self$scales, L$sc)
                     },
                     get_draws = function(keep=1:self$last_draw) {
                         " Returns matrix of draws, with iteration id and variable name as row and column names, respectively."
                         ## keep is a vector of draw indices of draws to keep.
                         ## defaults to all
                         D <- self$draws[keep,]
                         dimnames(D) <- list(iteration=self$iter[keep],
                                             variable=self$varnames)
                         return(D)
                     }
                     ),
                 private=list(
                     adapt = function(ix) {
                         b <- min(self$step/ix,0.75)
                         mu.n <- self$mu + b * (self$x-self$mu)
                         mu.1 <- mala_p23(mu.n, self$A1)
                         G.n <- self$G + b * (crossprod(self$x-self$mu)-self$G)
                         G.1 <- mala_p23(G.n, self$A1)
                         sig.n <- self$sig + b * (exp(self$log.r)-self$opt_acc)
                         sig.1 <- mala_p1(sig.n, self$e1, self$A1)
                         self$mu <- mu.1
                         self$G <- G.1
                         self$sig <- sig.1
                         self$Lam <- self$G + self$e2*diag(self$nvars)
                     },
                     report = function(i) {
                         cat("iter ",i+self$last_iter,"\n")
                     },
                     iterate_batch = function(n.iter, n.thin=1, report_freq=1) {
                         d <- 0
                         nd <- floor(n.iter/n.thin)
                         M <- matrix(NA, nd, self$nvars)
                         lp <- rep(NA, nd)
                         idx <- rep(NA, nd)
                         a <- rep(NA,n.iter)
                         sc <- rep(NA, n.iter)
                         for (i in 1:n.iter) {
                             if ((i+self$last_iter) %% report_freq == 0) {
                                 private$report(i)
                             }
                             a[i] <- self$sample()
                             sc[i] <- self$sig
                             private$adapt(self$last_iter+i)
                             if (i %% n.thin == 0) {
                                 d <- d+1
                                 M[d,] <- self$x
                                 lp[d] <- self$log.fx
                                 idx[d] <- self$last_iter + i
                             }
                         }
                         return(list(iter=idx, dr=M, logpost=lp,
                                     acc=a, sc=sc))
                     }
                     )
                 )





