#include <RcppEigen.h>

#include <MVN_headers.h>

using Rcpp::Function;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::IntegerVector;
using Rcpp::Rcout;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using Rcpp::as;
using Rcpp::Named;
using Rcpp::List;
using Rcpp::_;
using Rcpp::wrap;




//[[Rcpp::export]]
List langMH(NumericVector x,
	    Function fn, Function gr,
	    NumericMatrix prCov,
	    NumericMatrix prChol,
	    int ndraws) {

  int k = x.size();
  NumericMatrix draws(ndraws, k);
  IntegerVector acc(ndraws);
  NumericVector logpost(ndraws);


  NumericVector y(k);
  NumericVector grx(k);
  NumericVector gry(k);
  Rcpp::NumericVector mx(k);
  NumericVector my(k);

  Map<VectorXd> xe(as<Map<VectorXd> >(x));
 
  Map<VectorXd> grxe(as<Map<VectorXd> >(grx));
  Map<VectorXd> grye(as<Map<VectorXd> >(gry));
  Map<VectorXd> mxe(as<Map<VectorXd> >(mx));
  Map<VectorXd> mye(as<Map<VectorXd> >(my));
 
  Map<MatrixXd> prCove = MatrixXd::Map(prCov.begin(), k, k);
  Map<MatrixXd> prChole = MatrixXd::Map(prChol.begin(), k, k);

  

  double log_fx = as<double>(fn(x));
  double log_fy;
  NumericVector log_px(1);
  NumericVector log_py(1);
  grx = as<NumericVector>(gr(x));

  
  for (int i=0; i<ndraws; i++) {

    if (remainder(i,10) == 0) {
      Rcout << "iter " << i << "\t";
    }
    mxe = xe + 0.5 * prCove * grxe; // defines mx
    y = rMVN_vec(mx, prChol, false);
    Map<VectorXd> ye(as<Map<VectorXd> >(y));   
    log_py = dMVN_(y, mx, prChol, false);
    log_fy = as<double>(fn(y));

    gry = as<NumericVector>(gr(y));
    mye = ye + 0.5 * prCove * grye; // defines my
    log_px = dMVN_(x, my, prChol, false);

    double log_r = fmin(log_fy - log_fx + log_px(0) - log_py(0), 0);
    double log_u = log(as<double>(Rcpp::runif(1)));

    if (log_u <- log_r) { // accept
      if (remainder(i,10) == 0) {
	Rcout << "ACCEPT\n";
      }
      x = y;
      log_fx = log_fy;
      grx = gry;
      acc(i) = 1;
    } else {
      if (remainder(i,10) == 0) {
	Rcout << "REJECT\n";
      }
      acc(i) = 0;
    }
    draws(i,_) = x;
    logpost(i) = log_fx;
  }

  List res = List::create(Named("draws")=wrap(draws),
			  Named("logpost") = wrap(logpost),
			  Named("acc") = wrap(acc));
  return(res);


}





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


