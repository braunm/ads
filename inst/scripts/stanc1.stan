## stan1c.stan - model for creatives as a poisson


data {

    int T;
    int J;		// number of brands sold
    int Jb;		// number of brands that advertised
    int JbE;		// length of number of non zero E columns
    matrix[T,Jb] A;
    int E[T,JbE];
}


parameters {
vector[JbE] gl0;
}

model {

    vector[JbE] ldE;

for(t in 1:T) {

    for(j in 1:JbE) ldE[j] <- exp(gl0[j]);
    for(j in 1:JbE) E[t,j] ~ poisson(ldE[j]);

	}
}
