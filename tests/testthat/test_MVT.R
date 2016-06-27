context("MVT")
test_that("MVT", {

    require(mnormt)

    N <- 2
    k <- 250
    df <- 4
    mu <- seq(-3,3,length=k)
    S <- rWishart(1,k+5,diag(k))[,,1] ## covariance
    P <- solve(S)
    chol.S <- t(chol(S))
    chol.P <- t(chol(P))

    set.seed(123)
    t1 <- system.time(x1 <- rmt(N,mean=mu, df=df, S=S))
    q1 <- system.time(d1 <- dmt(x1,mean=mu, S=S, df=df, log=TRUE))

    q2a <- system.time(d2a <- dMVT(x1, mu, chol.S, df, FALSE))
    q2b <- system.time(d2b <- dMVT(x1, mu, chol.P, df, TRUE))

    expect_equal(d1,d2a)
    expect_equal(d1,d2b)
    expect_equal(d2a, d2b)

    set.seed(123)
    t2a <- system.time(x2a <- rMVT(N, mu, chol.S, df, FALSE))
    t2b <- system.time(x2b <- rMVT(N, mu, chol.P, df, TRUE))


})
