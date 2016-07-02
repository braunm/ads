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
#' @field x current state of the sampler
#' @field log.fx log f(x)
#' @field grx gradient at x
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
                     initialize = function(start, DL, id, seed, n.thin) {
                         "Initializing function"
                         self$DL <- DL
                         self$id <- id
                         self$seed <- seed
                         private$cl <- new("ads",DL)
                         cl$initialize(start)
                         self$nvars <- length(start)
                         self$x <- start
                         self$log.fx <- cl$get.f(start)
                         self$grx <- cl$get.df(start)
                         private$draws <- NULL
                         private$logpost <- NULL
                         private$acc <- NULL
                         private$n.thin <- n.thin
                         set.seed(self$seed)
                         private$prCov <- matrix(NA,self$nvars, self$nvars)
                         private$prChol <- t(chol(private$prCov))
                     },
                     sample = function() {
                         mx <- t(self$x) + 0.5 * prCov %*% self$grx
                         y <- rMVN(1, mx, private$prChol, FALSE)
                         log.py <- dMVN(y, mx, private$prChol, FALSE)
                         log.fy <- cl$get.f(y)

                         gry <- cl$get.df(y)
                         my <- t(y) + 0.5 * prCov %*% gry
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
                     iterate = function(n, x=self$x) {
                         L <- iterate_batch(n)
                         last_iter <- last_iter + 1
                         last_draw <- last_draw + n/private$n.thin
                         private$iter <- c(private$iter, L$iter)
                         private$draws <- rbind(private$draws,L$dr)
                         private$logpost <- c(private$logpost, L$logpost)
                         private$acc <- c(private$acc, L$acc)
                         self$acc_rate <- mean(private$acc)
                     }
                     ),
                 private=list(
                     cl = NA,
                     draws = NULL,
                     prCov = NA,
                     prChol = NA,
                     iter = NULL,
                     logpost = NULL,
                     last_iter = 0,
                     last_draw = 0,
                     n.thin = 1,
                     iterate_batch = function(n.iter) {
                         d <- 0
                         nd <- floor(n.iter/private$n.thin)
                         M <- matrix(NA, nd, self$nvars)
                         logpost <- rep(NA, nd)
                         idx <- rep(NA, nd)
                         a <- rep(NA,n.iter)
                         for (i in 1:ndraws) {
                             acc[i] <- sample()
                             if (i %% private$n.thin == 0) {
                                 d <- d+1
                                 M[d,] <- self$x
                                 logpost[d] <- self$log.fx
                                 idx[d] <- i
                             }

                         }
                         return(list(iter=idx, dr=M, logpost=logpost,
                                     acc=a))
                     }
                     )
                 )





