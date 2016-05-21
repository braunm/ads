## stan7.stan - DLM 1: matrix normals (specific to advertising paper)
## fits model
## Yt = F1_t \Theta_{11t} + \Beta X + v_{1t}, V1 ~ N(0,V1,Sigma)
## \Theta_{11t} = F2_t \Theta_{2t} + v_{2t}, V2 ~ N(0,V2,Sigma)
## \Theta_{2t} = G_t \Theta_{2t-1} + H_t + w_t
## H_t = H_{0t} + Phi H_

## Need to specify this by stating elements F1t, F2t, Gt, Ht, X, V1, V2, and W,
## Sigma is given a prior IW(nu0, Omega0) and the state variables are integrated out
## forming a matrix T from the above system.

## This version allow for J brands sales, with 1:Jb of those brands to be advertised
## and 1:JbE to have positive creative (E>0 for all weeks). They must be organized in that way.

data {

    int N;
    int T;
    int J;		// number of brands sold
    int Jb;		// number of brands that advertised
    int K;
    int P;
    int JbE;		// length of number of non zero E columns
    matrix[N, J] Y[T];
    matrix[N, K] X[T];
    matrix[N, N*(1+P)] F1[T];
	matrix[N*(1+P),(1+Jb+P)] F2[T];
    
    matrix[N, 1+Jb+P] F1F2[T]; 
    matrix[T,Jb] A;
    int E[T,JbE];
    real<lower=J> nu0;
    cov_matrix[J] Omega0;
    matrix[1+Jb+P,J] M20;
    cov_matrix[1+Jb+P] C20;
}

transformed data {

    matrix[T,Jb] lA;
    int V1_dim;
    int V2_dim;

    int W1_dim;
    int W2_dim;
    int W_dim;

    V1_dim <- N;
    V2_dim <- N*(1+P);
    W1_dim <- Jb+1;
    W2_dim <- P;
    W_dim <- W1_dim + W2_dim;

    for(t in 1:T) for(j in 1:Jb) lA[t,j] <- log1p(A[t,j]);

}

parameters {
    matrix[K,J] theta12;
    matrix[JbE,J] phi;
    real<lower=0, upper=1> delta;
 vector<lower=0>[W_dim] Wd;
 vector<lower=0>[V1_dim] V1d;
  vector<lower=0>[V2_dim] V2d;
// creative additions;
vector[JbE] gl0;
vector[J] gl1[JbE];
// advertising allocation;
vector[Jb] ga0;
vector[J] ga1[Jb];
    real<lower=0> sa[Jb];
matrix[1+Jb+P,J] M2t[T+1];
matrix[1+Jb+P,1+Jb+P] C2t[T+1];

}

model {

    matrix[1+Jb+P, 1+Jb+P] Gt; 
    matrix[N,N] Qt;
    matrix[1+Jb+P,1+Jb+P] R2t;
    matrix[N*(1+P),N*(1+P)] R1t;
    matrix[1+Jb+P, N] S2t;
    matrix[N, J] Yft;
    matrix[1+Jb+P, J] Ht;
    matrix[1+Jb+P, J] a2t;
    matrix[N,J] Ybar;
    real log_det_Qt;
    matrix[J,J] OmegaT;
    //vector[JbE] ldE;

    C2t[1] <- C20;
    M2t[1] <- M20;
    OmegaT <- Omega0;

    Gt <- diag_matrix(rep_vector(1,1+Jb+P));
    Ht <- rep_matrix(0, 1+Jb+P,J);
    log_det_Qt <- 0;

    Gt[1,1] <- 1-delta;

	Wd ~ normal(0, 10); 				// prior on diagonal for W
	V1d ~ normal(0, 100);
	V2d ~ normal(0, 100);
   //     for(j in 1:JbE) phi[j] ~ normal(0,10);

	// sa ~ normal(0,100);

for(t in 1:T) {

	// advertising
    // for(j in 1:Jb) lA[t,j] ~ normal(ga0[j] + row(M2t[t],1) * ga1[j], sa[j]);

	// creative additions
    // for(j in 1:JbE) ldE[j] <- exp(gl0[j] + row(M2t[t],1+j) * gl1[j]);

    // for(j in 1:JbE) E[t,j] ~ poisson(ldE[j]);

    for (j in 1:Jb) Gt[1,j+1] <- lA[t,j];				// the first 1:Jb have advertising
    for(j in 1:JbE) for(k in 1:J) Ht[1+j, k] <- phi[j, k] * E[t, j]; 	// the first 1:JbE have creatives
    
    a2t <- Gt * M2t[t] + Ht;
    Ybar <- Y[t] - X[t] * theta12;
    Yft <- Ybar - F1F2[t] * a2t;
    R2t <- diag_matrix(Wd) + quad_form_sym(C2t, Gt');
    R1t <- diag_matrix(V2d) + quad_form_sym(R2t,F2[t]');
    Qt <- diag_matrix(V1d) + quad_form_sym(R1t, F1[t]');

    increment_log_prob(-0.5*J*log_determinant(Qt));

    OmegaT <- OmegaT + quad_form_sym(inverse_spd(Qt), Yft);

    S2t <- R2t * F1F2[t]';
    M2t[t+1] <- a2t + S2t/Qt * Yft;
    C2t[t+1] <- R2t - S2t/Qt * S2t';

}

    increment_log_prob(-0.5*(nu0 + T * N) * log_determinant(OmegaT));

}

generated quantities {
    matrix[1+Jb+P, 1+Jb+P] Gt; 
    matrix[1+Jb+P,J] M2t[T+1];

    for (j in 1:Jb) Gt[1,j+1] <- lA[t,j];				// the first 1:Jb have advertising
    for(j in 1:JbE) for(k in 1:J) Ht[1+j, k] <- phi[j, k] * E[t, j]; 	// the first 1:JbE have creatives

}