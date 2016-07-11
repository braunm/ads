#include <RcppEigen.h>

// #include <MVN_headers.h>


using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::IntegerVector;
using Rcpp::Rcout;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using Eigen::Lower;
using Rcpp::as;
using Rcpp::Named;
using Rcpp::List;
using Rcpp::_;
using Rcpp::wrap;


//' @title dMVN
//' @name dMVN
//' @param X_ matrix
//' @param mu_ vector
//' @param L_ lower chol of cov or prec matrix
//' @param isPrec covariance or precision matrix?
//' @return Numeric vector
//' @export
//[[Rcpp::export]]
NumericVector dMVN(NumericMatrix X_, NumericVector mu_,
		   NumericMatrix L_, bool isPrec){


  size_t k = X_.cols();
  size_t N = X_.rows();
  size_t q = mu_.size();
  //  assert(k == mu_.rows());

  Map<MatrixXd> X = MatrixXd::Map(X_.begin(), N, k);
  Map<VectorXd> mu = VectorXd::Map(mu_.begin(), k);
  Map<MatrixXd> L = MatrixXd::Map(L_.begin(), k, k);
  double C = -0.918938533204672669541 * k; // -k*log(2*pi)/2
  double detL = L.diagonal().array().log().sum();

  MatrixXd xmu = X.transpose().colwise() - mu;
  VectorXd logdens(N);


  if (isPrec) {
    MatrixXd Z = L.triangularView<Lower>().transpose() * xmu;
    logdens.array() = C + detL - 0.5 * (Z.array() * Z.array()).colwise().sum();
  } else {
    MatrixXd Z = L.triangularView<Lower>().solve(xmu);  
    logdens.array() = C - detL - 0.5 * (Z.array() * Z.array()).colwise().sum();
  }
  return(wrap(logdens));
  
}


//' @title rMVN
//' @name rMVN
//' @param N integer, number of draws
//' @param mu_ mean vector
//' @param L_ lower chol of cov or prec matrix
//' @param isPrec covariance or precision matrix?
//' @return Numeric matrix
//' @export
//[[Rcpp::export]]
NumericMatrix rMVN(int N, NumericVector mu_,
		   NumericMatrix L_, bool isPrec){

  size_t k = mu_.size();
  NumericVector z_ = Rcpp::rnorm(N*k);
  Map<MatrixXd> Z = MatrixXd::Map(z_.begin(), k, N);
  Map<VectorXd> mu = VectorXd::Map(mu_.begin(), k);
  Map<MatrixXd> L = MatrixXd::Map(L_.begin(), k, k);
  MatrixXd X;
  
  if (isPrec) {
    X = ((L.triangularView<Lower>().transpose().solve(Z)).colwise() + mu).transpose();
  } else {
    X = ((L.triangularView<Lower>() * Z).colwise() + mu).transpose();
  }
  return(wrap(X));
 
}


