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
    matrix[J,J] phi;
    real<lower=0, upper=1> delta;
    cov_matrix V[V_dim, V_dim] V;
}

model {

    matrix[1+J+P, 1+J+P] Gt; ## CHECK TRANSPOSITION
    cov_matrix[N] Qt;  
    cov_matrix[1+J+P] R2t;
    matrix[1+J+P,J] M2t;
    cov_matrix[1+J+P] C2t;
    matrix[N, 1+J+P] S2t;
    matrix[N, J] Yft;
    matrix[J, J] Ht;
    matrix[1+J+P, J] a2t;
    matrix[N,J] Ybar;
    real log_det_Qt;

    for (i in 1:T) {

       

        a2t <- Gt * M2t;

        set_Ht();

        for (j in 1:J) {
            row(a2t, j+1) <- row(a2t, j+1) + row(Ht, j);
        }

        Ybar <- Y[i] - X[i] * theta12;
        Yft <- Ybar - F1F2[i]' * a2t;'
        R2t <- W + quad_form_sym(C2t, Gt);
        Qt <- V + quad_form_sym(R2t, F1F2[i]);

        log_det_Qt <- log_determinant(Qt);

        S2t <- R2t * F1F2[i];
        M2t <- a2t + (S2t/Qt)*Yft;
        C2t <- R2t - S2t
       
                                 
                                 



    }
    
    
    
    row_vector[J] yit;
    yit <- Y[1,5];
    print(yit);
        
    z ~ normal(0,1);

}
