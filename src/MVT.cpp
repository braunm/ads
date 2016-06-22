#include <RcppEigen.h>

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



//' @title dMVT
//' @param X_ matrix
//' @param mu_ vector
//' @param L_ lower chol of cov or prec matrix
//' @param v degrees of freedom (must be >=3)
//' @param isPrec covariance or precision matrix?
//' @return Numeric vector
//' @export
//[[Rcpp::export]]
NumericVector dMVT(NumericMatrix X_, NumericVector mu_,
		 NumericMatrix L_, double v, bool isPrec){


  size_t k = X_.cols();
  size_t N = X_.rows();
  size_t q = mu_.size();
 

  Map<MatrixXd> X = MatrixXd::Map(X_.begin(), N, k);
  Map<VectorXd> mu = VectorXd::Map(mu_.begin(), k);
  Map<MatrixXd> L = MatrixXd::Map(L_.begin(), k, k);
  double C = lgamma((v+q)/2) - lgamma(v/2) - q*(log(v)+log(M_PI))/2;
  //  double C = -0.918938533204672669541 * k; // -k*log(2*pi)/2
  double detL = L.diagonal().array().log().sum();

  MatrixXd xmu = X.transpose().colwise() - mu;
  VectorXd logdens(N);

  double c2;
  MatrixXd Z;
  if (isPrec) {
    Z = L.triangularView<Lower>().transpose() * xmu;
    c2 = C + detL;
  } else {
    Z = L.triangularView<Lower>().solve(xmu);  
    c2 = C - detL;
  }
  logdens.array() = c2 - 0.5*(v+q)*(1+(Z.array() * Z.array()).colwise().sum()/v).log();
  return(wrap(logdens));
  
}


//' @title rMVT
//' @param N integer, number of draws
//' @param mu_ mean vector
//' @param L_ lower chol of cov or prec matrix
//' @param v degrees of freedom (must be >=3)
//' @param isPrec covariance or precision matrix?
//' @return Numeric matrix
//' @export
//[[Rcpp::export]]
NumericMatrix rMVT(int N, NumericVector mu_,
		 NumericMatrix L_, double v, bool isPrec){

  size_t k = mu_.size();
  NumericVector z_ = Rcpp::rnorm(N*k);
  NumericVector u = Rcpp::rchisq(1,v);
  Map<MatrixXd> Z = MatrixXd::Map(z_.begin(), k, N);
  Map<VectorXd> mu = VectorXd::Map(mu_.begin(), k);
  Map<MatrixXd> L = MatrixXd::Map(L_.begin(), k, k);
  MatrixXd X;
  MatrixXd Y;
 
  
  if (isPrec) {
    Y = L.triangularView<Lower>().transpose().solve(Z);
    
  } else {
    Y = L.triangularView<Lower>() * Z;
  }
  
  X = ((sqrt(v/u(0))*Y).colwise() + mu).transpose();
  
  return(wrap(X));
 
}
