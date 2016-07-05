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
                     logpost = NULL,
                     last_iter = 0,
                     last_draw = 0,
                     varnames = NA,
                     initialize = function(start, DL, id, seed, hs.mode, sig) {
                         "Initializing function"
                         self$DL <- DL
                         self$id <- id
                         self$seed <- seed
                         private$cl <- new("ads",DL)
                         private$cl$record.tape(start)
                         self$nvars <- length(start)
                         self$x <- matrix(start,nrow=1)
                         self$log.fx <- private$cl$get.f(start)
                         self$grx <- private$cl$get.df(start)
                         self$draws <- NULL
                         self$logpost <- NULL
                         self$acc <- NULL
                         set.seed(self$seed)
                         private$prCov <- sig*solve(-hs.mode)
                         private$prChol <- t(chol(private$prCov))
                         self$last_iter <- 0
                         self$last_draw <- 0
                         self$varnames <- make_varnames(DL$dimensions, DL$flags)
                     },
                     sample = function() {
                         mx <- as.vector(self$x + 0.5 * self$grx %*% private$prCov)
                         y <- rMVN(1, mx, private$prChol, FALSE) ## row matrix
                         log.py <- dMVN(y, mx, private$prChol, FALSE)
                         log.fy <- private$cl$get.f(as.vector(y))

                         gry <- private$cl$get.df(as.vector(y)) ## vector
                ##         browser()
                         my <- as.vector(t(y) + 0.5 * private$prCov %*% gry)
                         log.px <- dMVN(self$x, my, private$prChol, FALSE)

                         log.r <- min(log.fy-self$log.fx + log.px-log.py, 0)
                         log.u <- log(runif(1))
                         if (log.u <= log.r) { ## accept
                             self$x <- y
                             self$log.fx <- log.fy
                             self$grx <- gry
                             acc <- TRUE
                         } else {
                             acc <- FALSE
                         }
                         return(acc)
                     },
                     iterate = function(n.iter, n.thin=1,...) {
                         L <- private$iterate_batch(n.iter, n.thin, ...)
                         self$last_iter <- self$last_iter + n.iter
                         self$last_draw <- self$last_draw + n.iter/n.thin
                         self$iter <- c(self$iter, L$iter)
                         self$draws <- rbind(self$draws,L$dr) ## not efficient
                         self$logpost <- c(self$logpost, L$logpost)
                         self$acc <- c(self$acc, L$acc)
                         self$acc_rate <- mean(self$acc)
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
                     cl = NA,
                     prCov = NA,
                     prChol = NA,

                     iterate_batch = function(n.iter, n.thin=1) {
                         d <- 0
                         nd <- floor(n.iter/n.thin)
                         M <- matrix(NA, nd, self$nvars)
                         lp <- rep(NA, nd)
                         idx <- rep(NA, nd)
                         a <- rep(NA,n.iter)
                         for (i in 1:n.iter) {
                             cat("iter ",self$last_iter+i,"\n")
                             a[i] <- self$sample()
                             if (i %% n.thin == 0) {
                                 d <- d+1
                                 M[d,] <- self$x
                                 lp[d] <- self$log.fx
                                 idx[d] <- self$last_iter + i
                             }
                         }
                         return(list(iter=idx, dr=M, logpost=lp,
                                     acc=a))
                     }
                     )
                 )





