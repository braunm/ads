## stan1c.stan - model for creatives as a poisson


data {

    int T;
    int J;		// number of brands sold
    int Jb;		// number of brands that advertised
    int JbE;		// length of number of non zero E columns
    int R;      // number of columns in each brand CM
    matrix[T,Jb] A;
    int E[T,JbE];
    matrix[T,JbE*R] CMl;
}


parameters {
    vector[JbE] gl0;
    vector[Jb*R] gl1[JbE];
}

model {

    vector[T] ldE[JbE];

    gl0 ~ normal(0,1000);

    for(j in 1:JbE) {
        gl1[j] ~ normal(0,1000);
        ldE[j] <- exp( gl0[j] + CMl * gl1[j] );
        E[1:T,j] ~ poisson(ldE[j]);
    }
}
