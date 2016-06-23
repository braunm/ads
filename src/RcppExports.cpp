// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// dMVN
NumericVector dMVN(NumericMatrix X_, NumericVector mu_, NumericMatrix L_, bool isPrec);
RcppExport SEXP ads_dMVN(SEXP X_SEXP, SEXP mu_SEXP, SEXP L_SEXP, SEXP isPrecSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< NumericMatrix >::type X_(X_SEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu_(mu_SEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type L_(L_SEXP);
    Rcpp::traits::input_parameter< bool >::type isPrec(isPrecSEXP);
    __result = Rcpp::wrap(dMVN(X_, mu_, L_, isPrec));
    return __result;
END_RCPP
}
// rMVN
NumericMatrix rMVN(int N, NumericVector mu_, NumericMatrix L_, bool isPrec);
RcppExport SEXP ads_rMVN(SEXP NSEXP, SEXP mu_SEXP, SEXP L_SEXP, SEXP isPrecSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu_(mu_SEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type L_(L_SEXP);
    Rcpp::traits::input_parameter< bool >::type isPrec(isPrecSEXP);
    __result = Rcpp::wrap(rMVN(N, mu_, L_, isPrec));
    return __result;
END_RCPP
}
// dMVT
NumericVector dMVT(NumericMatrix X_, NumericVector mu_, NumericMatrix L_, double v, bool isInv);
RcppExport SEXP ads_dMVT(SEXP X_SEXP, SEXP mu_SEXP, SEXP L_SEXP, SEXP vSEXP, SEXP isInvSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< NumericMatrix >::type X_(X_SEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu_(mu_SEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type L_(L_SEXP);
    Rcpp::traits::input_parameter< double >::type v(vSEXP);
    Rcpp::traits::input_parameter< bool >::type isInv(isInvSEXP);
    __result = Rcpp::wrap(dMVT(X_, mu_, L_, v, isInv));
    return __result;
END_RCPP
}
// rMVT
NumericMatrix rMVT(int N, NumericVector mu_, NumericMatrix L_, double v, bool isInv);
RcppExport SEXP ads_rMVT(SEXP NSEXP, SEXP mu_SEXP, SEXP L_SEXP, SEXP vSEXP, SEXP isInvSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu_(mu_SEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type L_(L_SEXP);
    Rcpp::traits::input_parameter< double >::type v(vSEXP);
    Rcpp::traits::input_parameter< bool >::type isInv(isInvSEXP);
    __result = Rcpp::wrap(rMVT(N, mu_, L_, v, isInv));
    return __result;
END_RCPP
}
