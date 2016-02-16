

#include <RcppEigen.h>
#include <Eigen/Eigen>

#include <MVN_.h>

using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::as;
using Rcpp::List;
using Rcpp::Named;
using Rcpp::wrap;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::MatrixBase;
using Rcpp::Rcout;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Lower;
using Rcpp::_;


//' @title dMVN
//' @param X_ matrix
//' @param mu_ vector
//' @param L_ lower chol of cov or prec matrix
//' @param isPrec covariance or precision matrix?
//' @return Numeric vector
//' @export
//[[Rcpp::export]]
NumericVector dMVN(NumericMatrix X_, NumericVector mu_,
		   NumericMatrix L_, bool isPrec){
	
  NumericVector res = dMVN_(X_, mu_, L_, isPrec);
  return(wrap(res));
 
}


//' @title rMVN
//' @param N integer, number of draws
//' @param mu_ mean vector
//' @param L_ lower chol of cov or prec matrix
//' @param isPrec covariance or precision matrix?
//' @return Numeric matrix
//' @export
//[[Rcpp::export]]
NumericMatrix rMVN(int N, NumericVector mu_,
		   NumericMatrix L_, bool isPrec){

  NumericMatrix res = rMVN_(N, mu_, L_, isPrec);
  return(wrap(res));

}

