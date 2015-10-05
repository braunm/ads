data{

    int N;
    int T;
    int J;
    int K;
    int P;
    matrix[N, J] Y[T];
    matrix[N, K] X[T];
 ##   real F1F2[N, 1+J+P, T];
 ##   real A[J,T];
 ##   real E[J,T];
    matrix[N, 1+J+P] F1F2[T];
    vector[J] A[T];
    vector[J] E[T];
}

parameters {

    real z;

}

model {
    row_vector[J] yit;
    yit <- Y[1,5];
    print(yit);
        
    z ~ normal(0,1);

}
