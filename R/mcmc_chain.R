chain <- R6Class("chain",
                 public=list(
                     DL = NA,
                     id = 1,
                     seed = id,
                     nvars = NA,
                     iter = 1,
                     initialize = function(start, DL, id, seed) {
                         self$DL <- DL
                         self$id <- id
                         self$seed <- seed
                         private$cl <- new("ads",DL)
                         cl$initialize(start)
                         self$nvars <- length(start)
                     },
                     sample = function() {
                         mx <- t(x) + 0.5 * prCov %*% grx
                         y <- rMVN(1, mx, prChol, FALSE)
                         log.py <- dMVN(y, mx, prChol, FALSE)
                         log.fy <- cl$get.f(y)

                         gry <- cl$get.df(y)
                         my <- t(y) + 0.5 * prCov %*% gry
                         log.px <- dMVN(x, my, prChol, FALSE)

                         log.r <- min(log.fy-log.fx + log.px-log.py, 0)
                         log.u <- log(runif(1))
                         if (log.u <= log.r) { ## accept
                             x <- y
                             log.fx <- log.fy
                             grx <- gry
                             if (i %% report == 0)   cat(paste0("ACCEPT, chain: ", thread.id,  "\n"))
                             acc <- TRUE
                         } else {
                             if (i %% report == 0)    cat(paste0("REJECT, chain: ", thread.id,"\n"))
                             acc <- FALSE
                         }
                         return(acc)
                     },
                     iterate_batch = function(n.draws, n.thin) {
                         d <- 0
                         nd <- floor(n.draws/n.thin)
                         dr <- matrix(NA,nd, nvars)
                         acc <- rep(NA,n.draws)
                         for (i in 1:ndraws) {
                             acc[i] <- sample()
                             if (i %% n.thin == 0) {
                                 d <- d+1
                                 dr[d,] <- self$x
                                 logpost[d] <- self$log.fx
                             }
                         }
                         return(list(dr=dr, logpost=logpost,
                                     acc=acc))
                     }
                     iterate = function(n, x=self$x, start.i=1, n.thin=1) {
                         L <- iterate_batch(n, n.thin)
                         nd <- floor(n/n.thin)
                     }
                     ),
                 private=list(
                     x = rep(NA,nvars),
                     cl = NA,
                     draws = NULL,
                     log.fx = NA,
                     grx = rep(NA,nvars),
                     log.r = NA,
                     log.u = NA,
                     prCov = matrix(NA,nvars, nvars),
                     prChol = t(chol(prCpv))
                     )
                 )





