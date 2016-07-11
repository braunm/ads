#' @name mala_adapt
#' @title functions for adaptive MALA
#' @param e1,e2,A1 constants
#' @param P parameter being adapted
mala_p1 <- function(P, e1, A1) {
    if (P < e1) {
        return(e1)
    } else {
        if (P > A1) {
            return(A1)
        } else {
            return(P)
        }
    }
}

#' @rdname mala_adapt
mala_p23 <- function(P, A1) {

    fr <- sqrt(sum(P^2))
    if (fr <= A1) {
        return(P)
    } else {
        return(P*A1/fr)
    }
}
