## Copyright (C) 2013-2015 Michael Braun
#' @title Sampling_mvn
#' @aliases dmvn rmvn
#' @description Efficient sampling and density calculation from a multivariate
#' normal,
#' when given a Cholesky decomposition of a covariance or precision matrix.
#'
#' @param n number of samples
#' @param x numeric matrix, where each row is an MVN sample.
#' @param mu mean (numeric vector)
#' @param L A lower triangular matrix of the Cholesky factorization of
#' either the precision (default) or covariance matrix.  See details.
#' @param prec If TRUE, L is the Cholesky decomposition of the precision
#' matrix.  If false, it is the decomposition for the covariance matrix.
#'
#' @section Details:
#' These functions uses sparse matrix operations to sample from, or compute the
#' log density of, a multivariate normal distribution  The user must compute
#' the Cholesky decomposition first.
#'
#' @examples
#'    require(Matrix)
#'    m <- 20
#'    p <- 2
#'    k <- 4
#'
#' ## build sample sparse covariance matrix
#'    Q1 <- tril(kronecker(Matrix(seq(0.1,p,length=p*p),p,p),diag(m)))
#'    Q2 <- cBind(Q1,Matrix(0,m*p,k))
#'    Q3 <- rBind(Q2,cBind(Matrix(rnorm(k*m*p),k,m*p),Diagonal(k)))
#'    V <- tcrossprod(Q3)
#'    CH <- Cholesky(V)
#'
#'    x <- rmvn(10,rep(0,p*m+k),CH, FALSE)
#'  ##  print(x)
#'
#'    y <- dmvn(x[1,],rep(0,p*m+k), CH, FALSE)
#'  ##  print(y)
#'
#' @export
rmvn <- function(n, mu, L, prec=TRUE) {

    k <- length(mu)

    if (!(k>0)) {
        stop("mu must have positive length")
    }

    if (!(n>0)) {
        stop("n must be positive")
    }

    if (!(k==dim(L)[1])) {
        stop("dimensions of mu and L do not conform")
    }

    if (!is.logical(prec)) {
        stop("prec must be either TRUE or FALSE")
    }
    L <- Matrix::tril(L)
    x <- rnorm(n*k)
    dim(x) <- c(k,n)

    if (prec) {
        y <- Matrix::solve(Matrix::t(L),x) ## L'y = x
    } else {
        y <- L %*% x
    }

    y <- y + mu

    return(t(y))

}

#' @rdname rmvn
#' @export
dmvn <- function(x, mu, L, prec=TRUE) {


    if (is.vector(x) | (is.atomic(x) & NCOL(x)==1)) {
        x <- matrix(x,nrow=1)
    }

    k <- length(mu)
    n <- NROW(x)

    if (!(k>0)) {
        stop("mu must have positive length")
    }

    if (!(k==dim(L)[1])) {
        stop("dimensions of mu and L do not conform")
    }

    if (k!=NCOL(x)) {
        stop("x must have same number of columns as the length of mu")
    }


    if (!is.logical(prec)) {
        stop("prec must be either TRUE or FALSE")
    }

    L <- Matrix::tril(L)
    detL <- sum(log(Matrix::diag(L)))
    C <- -0.918938533204672669541*k ## -k*log(2*pi)/2

    xmu <- t(x)-mu
    if (prec) {
        z <- Matrix::crossprod(L,xmu)  ## L' %*% (x-mu)
        log.dens <- C + detL - Matrix::colSums(z*z)/2
    } else {
        z <- Matrix::solve(L, xmu) ## Lz = x-mu
        log.dens <- C - detL - Matrix::colSums(z*z)/2
    }

    return(as.numeric(log.dens))
}

