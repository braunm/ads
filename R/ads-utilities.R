#' Draw samples from a matrix normal using Cholesky decomposition
#'
#' @param ndraws Number of draws
#' @param M matrix mean (must by dimension nrow(S) x nrow(C))
#' @param C covariance matrix representing columns
#' @param S covariance matrix representing rows
#' @return ndraws matrices each of dimension S x C of random normal distribution
#' @examples
#' rmvMN(1, C, S)
rmvMN <- function(ndraws, M = rep(0, nrow(S) * ncol(C)), C, S) {
    ## set.seed(153)
    L <- chol(S) %x% chol(C)
    z <- rnorm(length(M))
    if (length(M) == 1) {
        res <- matrix(t(L) %*% z, nrow = nrow(C), ncol = ncol(S))
    }  else {
        res <- matrix(as.vector(M) + t(L) %*% z, nrow = nrow(C), ncol = ncol(S))
    }
    return(res)
}
