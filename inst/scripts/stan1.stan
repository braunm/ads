data {

    int N;
    int T;
    int J;
    int K;
    int P;
    matrix[N, J] Y[T];
    matrix[N, K] X[T];
    matrix[1+J+P, N] F1F2[T]; ## CHECK TRANSPOSITION
    vector[J] A[T];
    vector[J] E[T];
    real<lower=J> nu;
    cov_matrix[J] Omega0;
    matrix[1+J+P,J] M20;
    cov_matrix[1+J+P] C20;
}

transformed data {

    int V_dim;
    int W1_dim;
    int W2_dim;
    int W_dim;

    V_dim <- N;
    W1_dim <- J+1;
    W2_dim <- P;
    W_dim <- W1_dim + W2_dim;
}

parameters {
    real z;
    matrix[K,J] theta12;
 ##   matrix[J,J] phi;
    real<lower=0, upper=1> delta;
    cov_matrix[V_dim] V;
    cov_matrix[W_dim] W;
}

model {

    matrix[1+J+P, 1+J+P] Gt; ## CHECK TRANSPOSITION
    matrix[N,N] Qt;
    matrix[1+J+P,1+J+P] R2t;
    matrix[1+J+P,J] M2t;
    matrix[1+J+P,1+J+P] C2t;
    matrix[N, 1+J+P] S2t;
    matrix[N, J] Yft;
    matrix[1+J+P, J] Ht;
    matrix[1+J+P, J] a2t;
    matrix[N,J] Ybar;
    matrix[J,J] OmegaT;
    real log_det_Qt;
    real log_det_OmegaT;
    real log_PY;

    C2t <- C20;
    M2t <- M20;
    Gt <- diag_matrix(rep_vector(1,1+J+P));
    Ht <- rep_matrix(0, 1+J+P,J);


    Gt[1,1] <- 1-delta;

    for (i in 1:T) {


        ## set Gt and Ht
        for (j in 1:J) {
            Gt[1,j+1] <- log1p(A[i,j]);
  ##          Ht[j+1] <- row(phi, j);
        }


        a2t <- Gt * M2t + Ht;

        Ybar <- Y[i] - X[i] * theta12;
        Yft <- Ybar - F1F2[i]' * a2t;
        R2t <- W + quad_form_sym(C2t, Gt);
        Qt <- V + quad_form_sym(R2t, F1F2[i]);

        log_det_Qt <- log_determinant(Qt);

        S2t <- F1F2[i]' * R2t;
        M2t <- a2t + (S2t'/Qt) * Yft;
        C2t <- R2t - (S2t'/Qt) * S2t;
        OmegaT <- OmegaT + quad_form_sym(Yft, inverse(Qt));

    }

    z ~ normal(0,1);

}
