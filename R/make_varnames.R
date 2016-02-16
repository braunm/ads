#' @title Make vector of variable names
#' @param dims Vector of dimensions from data list
#' @return vector of variable names
#' @details cr_names for parameters added at tail end of vector.
#' Use str_split_fixed to convert result into a data frame.
#' @export
make_varnames <- function(dims, cr_names=NULL) {

    N <- dims["N"]
    T <- dims["T"]
    J <- dims["J"]
    Jb <- dims["Jb"]
    K <- dims["K"]
    P <- dims["P"]
    nfact.V1 <- dims["nfact.V1"]
    nfact.V2 <- dims["nfact.V2"]
    nfact.W1 <- dims["nfact.W1"]
    nfact.W2 <- dims["nfact.W2"]


    ## theta12 K x J

    g <- expand.grid(1:K, 1:J)
    theta12_names <- str_c("theta12",g$Var1,g$Var2,sep=".")

    ## phi  Jb x J

    g <- expand.grid(1:Jb,1:J)
    phi_names <- str_c("phi",g$Var1,g$Var2,sep=".")

    ## logit.delta

    logitdelta_names <- "logit_delta"

    ## V1 log_diag and factors
    V1_log_diag_names <- str_c("V1_log_diag",1:N,sep=".")
    V1_fact_names <- NULL
    if (nfact.V1 > 0) {
        for (j in 1:nfact.V1) {
            V1_fact_names <- c(V1_fact_names,
                               str_c("V1_fact",j:N,j,sep="."))
        }
    }

        ## V2 log_diag and factors
    V2_log_diag_names <- str_c("V2_log_diag",1:(N*(1+P)),
                               sep=".")
    V2_fact_names <- NULL
    if (nfact.V2 > 0) {
        for (j in 1:nfact.V2) {
            V2_fact_names <- c(V2_fact_names,
                               str_c("V2_fact",j:(N*(1+P)),
                                     j,sep="."))
        }
    }

        ## W1 log_diag and factors
    W1_log_diag_names <- str_c("W1_log_diag",1:(Jb+1),sep=".")
    W1_fact_names <- NULL
    if (nfact.W1 > 0) {
        for (j in 1:nfact.W1) {
            W1_fact_names <- c(W1_fact_names,
                               str_c("W1_fact",j:(Jb+1),
                                     j,sep="."))
        }
    }

        ## W2 log_diag and factors
    W2_log_diag_names <- str_c("W2_log_diag",1:P,sep=".")
    W2_fact_names <- NULL
    if (nfact.W2 > 0) {
        for (j in 1:nfact.W2) {
            W2_fact_names <- c(W2_fact_names,
                               str_c("W2_fact",j:P,j,sep="."))
        }
    }

    res <- c(theta12_names,
             phi_names,
             logitdelta_names,
             V1_log_diag_names,
             V1_fact_names,
             V2_log_diag_names,
             V2_fact_names,
             W1_log_diag_names,
             W1_fact_names,
             W2_log_diag_names,
             W2_fact_names,
             cr_names
             )

    return(res)

}

