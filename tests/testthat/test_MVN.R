context("MVN")
test_that("MVN", {


    require(mnormt)
  ##  set.seed(123)
    N <- 1500
    k <- 6
    mu <- seq(-3,3,length=k)
    S <- rWishart(1,k+5,diag(k))[,,1]
    P <- solve(S)
    chol.S <- t(chol(S))
    chol.P <- t(chol(P))

    x1 <- rmnorm(N,mean=mu,sqrt=t(chol.S))
    d1 <- dmnorm(x1,mean=mu,varcov=S,log=TRUE)
    d2a <- dMVN(x1,mu,chol.S,FALSE)
    d2b <- dMVN(x1,mu,chol.P,TRUE)
    expect_equal(d1,d2a)
    expect_equal(d1,d2b)
    expect_equal(d2a, d2b)

    ## set.seed(123)
    ## x2 <- rMVN(N, mu, chol.S, FALSE)
    ## browser()
    ## print(x2)
})
