## stan1.stan - DLM 1: matrix normals (specific to advertising paper)
## fits model
## Yt = F1_t \Theta_{11t} + \Beta X + v_{1t}, V1 ~ N(0,V1,Sigma)
## \Theta_{11t} = F2_t \Theta_{2t} + v_{2t}, V2 ~ N(0,V2,Sigma)
## \Theta_{2t} = G_t \Theta_{2t-1} + H_t + w_t
## H_t = H_{0t} + Phi H_

## Need to specify this by stating elements F1t, F2t, Gt, Ht, X, V1, V2, and W,
## Sigma is given a prior IW(nu0, Omega0) and the state variables are integrated out
## forming a matrix T from the above system.

data {

    int N;                          // number of cities (or rows of matrix outcome per time period)
    int T;                          // number of time periods
    int J;                          // number of brands (or columns of matrix outcome per time period)
    int K;                          // number of non time varying state parameters per brand
    int P;                          // number of time varying state parameters per brand
    matrix[N, J] Y[T];              // T elements of the matrix outcome (sales)
    matrix[N, K] X[T];              // T elements of non time varying covariates
    matrix[N, N*(1+P)] F1[T];       // T elements of the F1 matrix
    matrix[N*(1+P),(1+J+P)] F2[T];  // Matrix for hierarchical component

    matrix[N, 1+J+P] F1F2[T]; #     // Could be done in transformed block but is F1t x F2t matrix multiplication

    matrix[T,J] A;                  // Advertising
    vector[J] E[T];                 // Innovations due to new creatives
    real<lower=J> nu0;              // Prior scale for Sigma
    cov_matrix[J] Omega0;           // Prior covariance for Sigma
    matrix[1+J+P,J] M20;            // Initial prior on state mean
    cov_matrix[1+J+P] C20;          // Initial precision (uncertainty) on this state mean
}


transformed data {

    int V1_dim;                     // Corresponding covariance matrixes dimensions (square)
    int V2_dim;

    int W1_dim;
    int W2_dim;
    int W_dim;

    V1_dim <- N;
    V2_dim <- N*(1+P);
    W1_dim <- J+1;
    W2_dim <- P;
    W_dim <- W1_dim + W2_dim;
}

parameters {
    matrix[K,J] theta12;            // For time invariant component
    matrix[J,J] phi;                // For innovation component (creatives)
    real<lower=0,upper=1> delta;   // Decay parameter in advertising
    vector<lower=0>[W_dim] Wd;      // Diagonal for covariance for system evolution
    vector<lower=0>[V1_dim] V1d;    // Diagonal for city level covariance           (all independent)
    vector<lower=0>[V2_dim] V2d;    // Diagonal for parameter level city covariance (all independent)
}

model {

    // Declarations

    matrix[N,J] Ybar;               // Transformed outcome variable by subtracting time invariant component
    matrix[1+J+P, 1+J+P] Gt;        // System evolution matrix to be specified in model
    matrix[1+J+P, J] Ht;            // Innovation component to be specified in model

    // Below are intermediate values for matrix T

    matrix[N, J] Yft;
    matrix[N,N] Qt;
    matrix[N*(1+P),N*(1+P)] R1t;
    matrix[1+J+P,1+J+P] R2t;
    matrix[1+J+P,J] M2t;
    matrix[1+J+P,1+J+P] C2t;
    matrix[1+J+P, N] S2t;

    matrix[1+J+P, J] a2t;

    // The following are accumulated values

    matrix[J,J] OmegaT;
    real log_det_Qt;


    // Initialize recursion

    C2t <- C20;
    M2t <- M20;
    OmegaT <- Omega0;

    Gt <- diag_matrix(rep_vector(1,1+J+P));
    Ht <- rep_matrix(0, 1+J+P,J);
    log_det_Qt <- 0;

    delta ~ beta(1,3);
    Gt[1,1] <- 1-delta;

    // Set priors

    Wd ~ normal(0, 1000);
    V1d ~ normal(0, 1000);
    V2d ~ normal(0, 1000);

    // Begin recursion for matrix T

    for(t in 1:T) {
        for (j in 1:J) {
            Gt[1,j+1] <- log1p(A[t,j]);
            for(k in 1:J) Ht[j+1,k] <- phi[j,k]*E[t,k];     // k row of phi corresponds to k brand adv creative changed
        }
    a2t <- Gt * M2t + Ht;
    Ybar <- Y[t] - X[t] * theta12;
    Yft <- Ybar - F1F2[t] * a2t;
    R2t <- diag_matrix(Wd) + quad_form_sym(C2t, Gt');
    R1t <- diag_matrix(V2d) + quad_form_sym(R2t,F2[t]');
    Qt <- diag_matrix(V1d) + quad_form_sym(R1t, F1[t]');

    increment_log_prob(-0.5*J*log_determinant(Qt));

    OmegaT <- OmegaT + quad_form_sym(inverse_spd(Qt), Yft);

    // Used for next time period
    S2t <- R2t * F1F2[t]';
    M2t <- a2t + S2t/Qt * Yft;
    C2t <- R2t - S2t/Qt * S2t';
    }
    // Take determinant and add to log probability
    increment_log_prob(-0.5*(nu0 + T * N) * log_determinant(OmegaT));

}
