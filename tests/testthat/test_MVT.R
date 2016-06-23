context("MVT")
test_that("MVT", {

    require(mnormt)

    N <- 50000
    k <- 6
    df <- 4
    mu <- seq(-3,3,length=k)
    S <- rWishart(1,k+5,diag(k))[,,1] ## covariance
    P <- solve(S)
    chol.S <- t(chol(S))
    chol.P <- t(chol(P))

    set.seed(123)
    x1 <- rmt(N,mean=mu, df=df, S=S)
    d1 <- dmt(x1,mean=mu, S=S, df=df, log=TRUE)

    d2a <- dMVT(x1, mu, chol.S, df, FALSE)
    d2b <- dMVT(x1, mu, chol.P, df, TRUE)

    expect_equal(d1,d2a)
    expect_equal(d1,d2b)
    expect_equal(d2a, d2b)

    ## set.seed(123)
    ## x2a <- rMVT(N, mu, chol.S, df, FALSE)
    ## x2b <- rMVT(N, mu, chol.P, df, TRUE)

    ## browser()
    ## print(1)

})
