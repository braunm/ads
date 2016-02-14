
#ifndef __ADS_MVN
#define __ADS_MVN


#include <RcppEigen.h>
#include <Eigen/Eigen>


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



//' @title dMVN
//' @param X_ matrix
//' @param mu_ matrix
//' @param G_ full covariance or precision matrix
//' @param isPrec covariance or precision matrix?
//' @return Numeric vector
//' @export
//[[Rcpp::export]]
NumericVector dMVN(NumericMatrix X_, NumericVector mu_,
		   NumericMatrix L_, bool isPrec){
	

  size_t k = X_.rows();
  size_t N = X_.cols();
  size_t q = mu_.size();
  //  assert(k == mu_.rows());

  Map<MatrixXd> X = MatrixXd::Map(X_.begin(), N, k);
  Map<VectorXd> mu = VectorXd::Map(mu_.begin(), k);
  Map<MatrixXd> L = MatrixXd::Map(L_.begin(), k, k);
  double C = -0.918938533204672669541*k; // -k*log(2*pi)/2
  double detL = L.diagonal().array().log().sum();

  MatrixXd xmu = X.colwise() - mu;
  MatrixXd Z(k, N);
  VectorXd logdens(N);

  if (isPrec) {
    Z = L.triangularView<Lower>().transpose() * X;
    logdens = -(Z.array() * Z.array()).colwise().sum();
    logdens += VectorXd::Constant(N, C + detL);
  } else {
    Z = L.triangularView<Lower>().solve(xmu);
    logdens = -(Z.array() * Z.array()).colwise().sum();
    logdens += VectorXd::Constant(N,C - detL);
  }

  return(wrap(logdens));

  // NumericVector res(N);
  // for (size_t i=0; i<N; i++)
  //   res(i) = Value(out(i));
  
  // return(res);
}


#endif
