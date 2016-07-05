#' @title Make vector of variable names
#' @param dims Vector of dimensions from data list
#' @param flags details of the model
#' @return vector of variable names
#' @details cr_names for parameters added at tail end of vector.
#' Use str_split_fixed to convert result into a data frame.
#' @export
make_varnames <- function(dims, flags, cr_names=NULL) {

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
    if (flags$include.X) {
        g <- expand.grid(1:K, 1:J)
        theta12_names <- str_c("theta12",g$Var1,g$Var2,sep=".")
    } else {
        theta12_names <- NULL
    }


    ## phi  Jb x J
    if (flags$full.phi) {
        g <- expand.grid(1:Jb,1:J)
        phi_names <- str_c("phi",g$Var1,g$Var2,sep=".")
    } else {
        if (flags$phi.re) {
            phi_names <- str_c("phi",1:(Jb+2),sep=".")
        } else {
            phi_names <- str_c("phi",1:Jb,sep=".")
        }
    }

    if (flags$estimate.M20) {
        g <- expand.grid(1:(1+P+Jb),1:J)
        M20_names <- str_c("M20",g$Var1,g$Var2,sep=".")
    } else {
        M20_names <- NULL
    }

    if (flags$estimate.C20) {
        C20_names <- str_c("C20",1:(1+P+Jb),sep=".") ## diagonal matrix
    } else {
        C20_names <- NULL
    }

    ## logit.delta

    logitdelta_names <- "logit_delta"

    ## V1 log_diag and factors
    if (flags$fix.V1) {
        V1_log_diag_names <- NULL
        V1_fact_names <- NULL
    } else {
        V1_log_diag_names <- str_c("V1_log_diag",1:N,sep=".")
        V1_fact_names <- NULL
        if (nfact.V1 > 0) {
            for (j in 1:nfact.V1) {
                V1_fact_names <- c(V1_fact_names,
                                   str_c("V1_fact",j:N,j,sep="."))
            }
        }
    }

    ## V2 log_diag and factors
    if (flags$fix.V2) {
        V2_log_diag_names <- NULL
        V2_fact_names <- NULL
    } else {
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
    }

    ## W1 log_diag and factors

    if (flags$fix.W) {
        W1_log_diag_names <- NULL
        W2_log_diag_names <- NULL
        W1_fact_names <- NULL
        W2_fact_names <- NULL
    } else {
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
    }

    if (flags$endog.A) {
        g12 <- expand.grid(1:Jb,1:(J+1))
        G1_names <- str_c("G1",g12$Var1,g12$Var2,sep=".")
        G2_names <- str_c("G2",g12$Var1,g12$Var2,sep=".")
        G3_names <- str_c("G3",1:Jb,sep=".")
    } else {
        G1_names <- NULL
        G2_names <- NULL
        G3_names <- NULL
    }

    if (flags$endog.E) {
        ge <- expand.grid(1:Jb,1:(J+1))
        H1_names <- str_c("H1",ge$Var1,ge$Var2,sep=".")
    } else {
        H1_names <- NULL
    }

    if (flags$use.cr.pars) {
        cr_names <- str_c("cr",R-1,sep=",")
    } else {
        cr_names <- NULL
    }

    res <- c(theta12_names,
             phi_names,
             M20_names,
             C20_names,
             logitdelta_names,
             V1_log_diag_names,
             V1_fact_names,
             V2_log_diag_names,
             V2_fact_names,
             W1_log_diag_names,
             W1_fact_names,
             W2_log_diag_names,
             W2_fact_names,
             G1_names,
             G2_names,
             G3_names,
             H1_names,
             cr_names
             )

    return(res)

}

