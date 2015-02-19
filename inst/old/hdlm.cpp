#ifndef __ads
#define __ads

// 

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <atomic/log1p_atomic.hpp>
#include <atomic/log1p_exp_atomic.hpp>
#include <atomic/lgamma_atomic.hpp>
#include <atomic/inv_logit_atomic.hpp>
#include <Distributions/mat_normAD.cpp>
#include <Distributions/betadist.cpp>
#include <Distributions/norm.cpp>
#include <assert.h>


using Eigen::Matrix;
using Eigen::Array;
using Eigen::MatrixBase;
using Eigen::Dynamic;
using Eigen::SparseMatrix;
using Eigen::MappedSparseMatrix;
using Eigen::LLT;
using Eigen::LDLT;
using Eigen::Map;
using Eigen::DiagonalMatrix;
using Eigen::Upper;
using Eigen::Lower;
using Eigen::VectorXi; 
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Block;
using CppAD::Atomic::lgamma;
using CppAD::Atomic::inv_logit;
using CppAD::sign;
using CppAD::Atomic::log1p;
using CppAD::Atomic::log1p_exp;
using Rcpp::Rcout;
using Rcpp::List;


// Headers from funcs.cpp
template<typename AScalar>
AScalar Afunc(const AScalar&, const AScalar&);

template<typename TA, typename TZ, typename TC, typename TU, typename TG>
void set_Gt(const MatrixBase<TA>&,
	    const MatrixBase<TZ>&,
	    const MatrixBase<TC>&,
	    const MatrixBase<TU>&,
	    const typename TC::Scalar&,
	    const typename TC::Scalar&,
	    const MatrixBase<TG>&); 

template<typename TZ, typename TE, typename TP, typename TM, typename TC, typename TO, typename TH>
void set_Ht(const MatrixBase<TZ>&,
	    const MatrixBase<TE>&,
	    const MatrixBase<TP>&,
	    const MatrixBase<TM>&,
	    const MatrixBase<TC>&,
	    const MatrixBase<TO>&,
	    const typename TP::Scalar&,
	    const typename TP::Scalar&,
	    const MatrixBase<TH>&);	    



// Functions used in this file
  
template<typename T>
bool my_finite(const T& x) {
  return( (abs(x) <= __DBL_MAX__ ) && ( x == x ) );
}


template<typename T>
void print_mat(const MatrixBase<T>& X){
  
  for (int i=0; i<X.rows(); i++) {
    for (int j=0; j<X.cols(); j++) {
           PrintFor("\t",X(i,j));
      //  cout << X(i,j) << "\t";
    }
    Rcout << "\n";
  }
}

template <typename TA, typename TB>
void log_MVgamma(const TA& v,
		 const int& k,
		 TB& mvg){
  
  mvg = k*(k-1)*LOG_PI_4;
  for (int j=1; j<=k; j++){
    mvg += lgamma(v + (1-j)/2.); 
  }
}



class ads {

  typedef int Index; // Tpars is the type of the parameter vector
  typedef double Scalar;
  typedef CppAD::AD<Scalar> AScalar;

  typedef Matrix<AScalar, Dynamic, Dynamic> MatrixXA;
  typedef Matrix<AScalar, Dynamic, 1> VectorXA;
  typedef Matrix<AScalar, 1, Dynamic> RowVectorXA;
  typedef SparseMatrix<AScalar, Eigen::RowMajor> SparseMatrixXA;
  typedef MappedSparseMatrix<double> MappedSparseXd;
  typedef Array<AScalar, Dynamic, Dynamic> ArrayXA;

 public:
  
  ads(const List&);


  AScalar eval_f(const Eigen::Ref<VectorXA>&);

  template<typename Tpars>
    void unwrap_params(const MatrixBase<Tpars>&);
    
  AScalar eval_LL();
  AScalar eval_hyperprior();

  AScalar eval_LL(const Eigen::Ref<VectorXA>&);
  AScalar eval_hyperprior(const Eigen::Ref<VectorXA>&);
  
  
protected:
  
  // data
  
  std::vector<MatrixXA> Y; // each is  N x J
  std::vector<MatrixXA> X; // each is N x K
  std::vector<SparseMatrixXA> F1; // each is N x N(1+P), row major
  std::vector<SparseMatrixXA> F2; // each is N(1+P) x (1+J+P), row major
  std::vector<SparseMatrixXA> F1F2; // each is N x (1+J+P), row major
  std::vector<VectorXA> A; // national advertising, J
  std::vector<MatrixXA> Ybar; // each is  N x J
  std::vector<VectorXA> AjIsZero; // 1 if A_j == 0, 0 otherwise
  std::vector<VectorXA> E; // number of new creatives added for each brand

  // priors
  
  MatrixXA M20; // 1+P+J x J
  MatrixXA C20; // 1+P+J x 1+P+J
  MatrixXA Omega0;
  AScalar nu0;

  MatrixXA mean_theta12;
  MatrixXA chol_cov_row_theta12;
  MatrixXA chol_cov_col_theta12;
  
  MatrixXA mean_phi;
  MatrixXA chol_cov_row_phi;
  MatrixXA chol_cov_col_phi;
  
  AScalar c_mean_pmean;
  AScalar c_mean_psd;
  AScalar c_sd_pmean;
  AScalar c_sd_psd;

  AScalar u_mean_pmean;
  AScalar u_mean_psd;
  AScalar u_sd_pmean;
  AScalar u_sd_psd;


  AScalar delta_a;
  AScalar delta_b;

  AScalar diag_scale_V1;
  AScalar diag_df_V1;
  AScalar fact_scale_V1;
  AScalar fact_df_V1;

  AScalar diag_scale_V2;
  AScalar diag_df_V2;
  AScalar fact_scale_V2;
  AScalar fact_df_V2;

  AScalar df_scale_W1;
  AScalar s_scale_W1;
  AScalar W1_eta;
  AScalar corr_W1_const; // normalizing const for lkj prior

  AScalar diag_scale_W2;
  AScalar diag_df_W2;
  AScalar fact_scale_W2;
  AScalar fact_df_W2;


  int J; // number of brands
  int N; // number of cities
  int T; // number of weeks
  int K; // covariates with stationary parameters
  int P; // covariates with nonstationary parameters
  int nfact_V1; // factors to estimate V1
  int nfact_V2; // factors to estimate V2
  int nfact_W2; // factors to estimate W2

  int V1_dim, V2_dim, W_dim, W1_dim, W2_dim;


  // parameters

  MatrixXA theta12;
  AScalar logit_delta;
  AScalar delta;
  AScalar c_mean;
  AScalar c_log_sd;
  AScalar c_sd;
  VectorXA c_off; // J copy wearout parameters offset
  AScalar u_mean;
  AScalar u_log_sd;
  AScalar u_sd;
  VectorXA u_off; // J ad wearout parameters offset
  MatrixXA phi; // J x J

  VectorXA V1_log_diag;
  VectorXA V2_log_diag;
  AScalar W1_scale;
  VectorXA W2_log_diag;

  MatrixXA V1; 
  MatrixXA V2; 
  MatrixXA W;  

  MatrixXA LV1; 
  MatrixXA LV2; 
  MatrixXA LW1;
  MatrixXA LW2;
  AScalar logdet_W1_corr; // for W1 before scaling
  AScalar log_W1_jac;

 
  // intermediate values

  MatrixXA Gt;
  MatrixXA Ht;

  MatrixXA a2t;
  MatrixXA Yft;
  MatrixXA Qt; 
  MatrixXA R1t; 
  MatrixXA R2t; 
  MatrixXA M2t;
  MatrixXA C2t;
  MatrixXA S2t; 
  MatrixXA OmegaT;
  MatrixXA Pneg;

  AScalar log_const;
  MatrixXA QYf;

  VectorXA c;
  VectorXA u;

  AScalar log_mvgamma_prior;
  AScalar log_mvgamma_post;

  AScalar zero;
  AScalar half;
  AScalar one;
  AScalar two;

  // flags for model specification
  bool include_H;
  bool add_prior;
  bool include_X;
  AScalar A_scale;

};



ads::ads(const List& params)
{

  using Rcpp::List;
  using Rcpp::NumericMatrix;
  using Rcpp::NumericVector;
  using Rcpp::IntegerVector;
  using Rcpp::as;
  using Eigen::MatrixXd;
  using Eigen::Map;
  using Rcpp::Rcout;
  using std::endl;

  const List & pars = static_cast<const List&>(const_cast<List &>(params));

  const List data = as<const List>(pars["data"]);
  const List priors = as<const List>(pars["priors"]);
  const List dimensions = as<const List>(pars["dimensions"]);
  const List flags = as<const List>(pars["flags"]);

  T = as<int>(dimensions["T"]);
  N = as<int>(dimensions["N"]);
  J = as<int>(dimensions["J"]);
  K = as<int>(dimensions["K"]);
  P = as<int>(dimensions["P"]);

  include_H = as<bool>(flags["include.H"]);
  add_prior = as<bool>(flags["add.prior"]);
  include_X = as<bool>(flags["include.X"]);
  A_scale = as<double>(flags["A.scale"]);

  List Xlist;
  if (include_X) {
    Xlist = as<List>(data["X"]);
    X.resize(T);
    assert(K>0);
  } else {
    assert(K==0);
  }

  const List Ylist = as<List>(data["Y"]);
  Y.resize(T);
  Ybar.resize(T);

  const List F1list = as<List>(data["F1"]);
  F1.resize(T);

  const List F2list = as<List>(data["F2"]);
  F2.resize(T);
  F1F2.resize(T);

  const List Alist = as<List>(data["A"]);
  A.resize(T);
  AjIsZero.resize(T);


  List Elist;
  if (include_H) {
    Elist = as<List>(data["E"]);
    E.resize(T);
  }

  // number of factors for estimating covariance matrices
  nfact_V1 = as<int>(dimensions["nfact.V1"]); 
  nfact_V2 = as<int>(dimensions["nfact.V2"]);
  if (P>0) {
    nfact_W2 = as<int>(dimensions["nfact.W2"]);
  } else {
    nfact_W2 = 0;
  }


  V1_dim = N;
  V2_dim = N*(P+1);
  W1_dim = 1+J;
  W2_dim = P;
  W_dim = W1_dim + W2_dim;

  Rcout << "Loading data\n";

  for (int i=0; i<T; i++) {

    if (include_X) {
      const Map<MatrixXd> Xd(as<Map<MatrixXd> >(Xlist[i]));
      X[i] = Xd.cast<AScalar>();
    }      

    const Map<MatrixXd> Yd(as<Map<MatrixXd> >(Ylist[i]));
    Y[i] = Yd.cast<AScalar>();

    const Map<VectorXd> Ad(as<Map<VectorXd> >(Alist[i]));
    A[i] = Ad.cast<AScalar>();

    AjIsZero[i].resize(J);
    for (int j=0; j<J; j++) {
      AjIsZero[i](j) = A[i](j)==0 ? 1. : 0.;
    }

    if (include_H) {

      const Map<VectorXd> Ed(as<Map<VectorXd> >(Elist[i]));
      E[i] = Ed.cast<AScalar>();
    }
 
    const MappedSparseXd F1d(as<MappedSparseXd >(F1list[i]));
    F1[i] = F1d.cast<AScalar>().transpose(); // transpose should force row major

    const MappedSparseXd F2d(as<MappedSparseXd >(F2list[i]));
    F2[i] = F2d.cast<AScalar>().transpose(); // transpose should force row major

    F1[i].makeCompressed();
    F2[i].makeCompressed(); 
    F1F2[i] = F1[i] * F2[i];


    Ybar[i].resize(N,J);
  }

  // These priors are required
  const Map<MatrixXd> M20_d(as<Map<MatrixXd> >(priors["M20"]));
  M20 = M20_d.cast<AScalar>();
 
  const Map<MatrixXd> C20_d(as<Map<MatrixXd> >(priors["C20"]));
  C20 = C20_d.cast<AScalar>();

  const Map<MatrixXd> Omega_d(as<Map<MatrixXd> >(priors["Omega0"]));
  Omega0 = Omega_d.cast<AScalar>();

  nu0 = as<double>(priors["nu0"]);
  log_MVgamma(nu0/2.,J,log_mvgamma_prior);
  log_MVgamma((nu0+T*N)/2.,J,log_mvgamma_post);
  LDLT<MatrixXA> chol_Omega0(Omega0);
  const AScalar log_det_Omega0 = chol_Omega0.vectorD().array().log().sum();
  log_const = -half*(N*T*J)*LOG_PI + log_mvgamma_prior;
  log_const -=  log_mvgamma_post + nu0*log_det_Omega0/2.;
  
  // The following priors are optional
  if (add_prior) {
    if (include_H) {
      const List priors_phi = as<List>(priors["phi"]);
      const Map<MatrixXd> mean_phi_d(as<Map<MatrixXd> >(priors_phi["mean"]));
      mean_phi = mean_phi_d.cast<AScalar>();
      const Map<MatrixXd> chol_cov_row_phi_d(as<Map<MatrixXd> >(priors_phi["chol.row"]));
      chol_cov_row_phi = chol_cov_row_phi_d.cast<AScalar>();
      const Map<MatrixXd> chol_cov_col_phi_d(as<Map<MatrixXd> >(priors_phi["chol.col"]));
      chol_cov_col_phi = chol_cov_col_phi_d.cast<AScalar>();
    }

    const List priors_delta = as<List>(priors["delta"]);
    delta_a = as<double>(priors_delta["a"]);
    delta_b = as<double>(priors_delta["b"]);
    
    const List priors_c = as<List>(priors["c"]);
    c_mean_pmean = as<double>(priors_c["mean.mean"]);
    c_mean_psd = as<double>(priors_c["mean.sd"]);
    c_sd_pmean = as<double>(priors_c["sd.mean"]);
    c_sd_psd = as<double>(priors_c["sd.sd"]);


    const List priors_u = as<List>(priors["u"]);
    u_mean_pmean = as<double>(priors_u["mean.mean"]);
    u_mean_psd = as<double>(priors_u["mean.sd"]);
    u_sd_pmean = as<double>(priors_u["sd.mean"]);
    u_sd_psd = as<double>(priors_u["sd.sd"]);

    if (include_X) {
      const List priors_theta12 = as<List>(priors["theta12"]);
      
      const Map<MatrixXd> mean_theta12_d(as<Map<MatrixXd> >(priors_theta12["mean"]));
      mean_theta12 = mean_theta12_d.cast<AScalar>();
      
      const Map<MatrixXd> chol_cov_row_theta12_d(as<Map<MatrixXd> >(priors_theta12["chol.row"]));
      
      chol_cov_row_theta12 = chol_cov_row_theta12_d.cast<AScalar>();
      
      const Map<MatrixXd> chol_cov_col_theta12_d(as<Map<MatrixXd> >(priors_theta12["chol.col"]));
      chol_cov_col_theta12 = chol_cov_col_theta12_d.cast<AScalar>();
    }


    const List priors_V1 = as<List>(priors["V1"]); 
    diag_scale_V1 = as<double>(priors_V1["diag.scale"]);
    diag_df_V1 = as<double>(priors_V1["diag.df"]);
    fact_scale_V1 = as<double>(priors_V1["fact.scale"]);
    fact_df_V1 = as<double>(priors_V1["fact.df"]);
    
    const List priors_V2 = as<List>(priors["V2"]); 
    diag_scale_V2 = as<double>(priors_V2["diag.scale"]);
    diag_df_V2 = as<double>(priors_V2["diag.df"]);
    fact_scale_V2 = as<double>(priors_V2["fact.scale"]);
    fact_df_V2 = as<double>(priors_V2["fact.df"]);

    const List priors_W1 = as<List>(priors["W1"]); 
    df_scale_W1 = as<double>(priors_W1["scale.df"]);
    s_scale_W1 = as<double>(priors_W1["scale.s"]);
    const double eta = as<double>(priors_W1["eta"]);
    W1_eta = eta;

    // from LKJ, Eq. 16
    double t1=0;
    double t2=0;
    for (int i=1; i<=(W1_dim-1); i++) {
      t1 += (2*eta-2+W1_dim)*(W1_dim-i);
      double tmp = eta+0.5*(W1_dim-i-1);
      t2 += (W1_dim-i)*(2*std::lgamma(tmp)-std::lgamma(2.0*tmp));
    }
    corr_W1_const = t1*LOG_2 + t2;

    if (P>0) {
      const List priors_W2 = as<List>(priors["W2"]); 
      diag_scale_W2 = as<double>(priors_W2["diag.scale"]);
      diag_df_W2 = as<double>(priors_W2["diag.df"]);
      fact_scale_W2 = as<double>(priors_W2["fact.scale"]);
      fact_df_W2 = as<double>(priors_W2["fact.df"]);
    }
  }
 
    
    // other priors go here, if needed.
    
    
    // Reserve space for parameters
  if (include_X) {
    theta12.resize(K,J);
  }
  
  // Reserve V1, V2 and W, and their factors
  V1.resize(V1_dim, V1_dim);
  V2.resize(V2_dim, V2_dim);
  W.resize(W_dim, W_dim);

  LV1.resize(V1_dim, nfact_V1);
  LV2.resize(V2_dim, nfact_V2);
  LW1.resize(W1_dim, W1_dim);

  V1_log_diag.resize(V1_dim);
  V2_log_diag.resize(V2_dim);
 
  if (P>0) {
    LW2.resize(W2_dim, W2_dim);
    W2_log_diag.resize(W2_dim);
  }
  
  c.resize(J);
  u.resize(J);
 
  if (include_H) {
    phi.resize(J,J);
    Ht = MatrixXA::Zero(J,J); // ignoring zeros in bottom P rows
  }
  
  Gt.resize(1+J+P,1+J+P);

  M2t.resize(1+J+P,J);
  C2t.resize(1+J+P,1+J+P);
  a2t.resize(1+J+P,J);
  Yft.resize(N,J);
  Qt.resize(N,N); 
  R1t.resize(N*(1+P),N*(1+P));
  R2t.resize(1+J+P,1+J+P);
  S2t.resize(1+J+P,N);
  OmegaT.resize(J,J);
  QYf.resize(N,J);

  if (include_H) {
    Pneg.resize(J,J);
  }

  zero = AScalar(0);
  half = AScalar(0.5);
  one = AScalar(1);
  two = AScalar(2);

  Rcout << "Constructor complete\n";

}

template<typename Tpars>
void ads::unwrap_params(const MatrixBase<Tpars>& par)
{

  // here we are copying the parameters.  Could we map instead? Doubtful because of the cast. 

  int ind = 0;
  
  // unwrap theta12 and construct Ybar

  if (include_X) {
    theta12 = MatrixXA::Map(par.derived().data()+ind,K,J);

    ind += K*J;
    for (int t=0; t<T; t++) {
      Ybar[t] = Y[t] - X[t] * theta12;
    }

  } else {
    for (int t=0; t<T; t++) {
      Ybar[t] = Y[t];
    }
  }

  // for c and u, the parameter is an offset agains the 
  // mean.  So c_j = c_mean + par[ind+j]

  c_mean = par(ind++); //ind increments after pull
  c_log_sd = par(ind++); // ind increments after pull
  c_sd = exp(c_log_sd);
  c_off = par.segment(ind,J); // N(0,1) prior
  ind += J;
  c.array() = c_sd*c_off.array() + c_mean;

 
  u_mean = par(ind++); //ind increments after pull
  u_log_sd = par(ind++); // ind increments after pull
  u_sd = exp(u_log_sd);
  u_off = par.segment(ind,J); // N(0,1) prior
  ind += J;
  u.array() = u_sd*u_off.array() + u_mean;

 
  if (include_H) {
    phi = MatrixXA::Map(par.derived().data()+ind,J,J);
    ind += J*J;
  }

  logit_delta = par(ind++);
  delta = inv_logit(logit_delta); 

  // Rcout << "logit_delta = " << logit_delta << "\n"; 


  // unwrap elements of V1, V2 and W, which are 
  // modeled as LL + S.  First, the log of diag(S)
  // is unwrapped. Then, if nfact>0, the unique elements of the 
  // factors, by column.  Diagonal elements are 
  // exponentiated for identification.

  V1.setZero();
  V1_log_diag = par.segment(ind,V1_dim);
  ind += V1_dim;
  V1.diagonal() = V1_log_diag.array().exp().matrix();

  //  Rcout << "\nV1.log.diag = " << V1_log_diag << "\n";

  if (nfact_V1 > 0) {
    LV1.setZero();
    for (int j=0; j<nfact_V1; j++) {
      LV1.block(j,j,V1_dim-j,1) = par.segment(ind,V1_dim-j);
      ind += V1_dim - j;
      LV1(j,j) = exp(LV1(j,j));      
    }
    V1.template selfadjointView<Eigen::Lower>().rankUpdate(LV1);
  }

  V2_log_diag = par.segment(ind,V2_dim);
  ind += V2_dim;
  V2.setZero();
  V2.diagonal() = V2_log_diag.array().exp().matrix();

  // Rcout << "V2.log.diag = " << V2_log_diag << "\n";

  if (nfact_V2 > 0) {
    LV2.setZero();
    for (int j=0; j<nfact_V2; j++) {
      LV2.block(j,j,V2_dim-j,1) = par.segment(ind,V2_dim-j);
      ind += V2_dim - j;
      LV2(j,j) = exp(LV2(j,j));
    }
    V2.template selfadjointView<Eigen::Lower>().rankUpdate(LV2);
  }

  // W is a scaled correlation matrix.
  // The log scale factor is the first
  // element.  Then there is a transform of the lower
  // triangle of the correlation matrix.
 

  W.setZero();
  W1_scale = exp(par(ind++));
  LW1.setZero();
  // copy terms to lower triangle
  logdet_W1_corr = 0;
  log_W1_jac = 0;
  for (int j=0; j<W1_dim-1; j++) {
    for (int i=j+1; i<W1_dim; i++) {
      LW1(i,j) = tanh(par(ind++));
      AScalar tmp = log1p(-pow(LW1(i,j),2));
      logdet_W1_corr += tmp;
      log_W1_jac += 0.5*(W1_dim-j)*tmp;
    }
  }

  Block<MatrixXA> W1 = W.topLeftCorner(1+J,1+J);
  W1(0,0)=1;
  W1.bottomLeftCorner(W1_dim-1,1) = LW1.bottomLeftCorner(W1_dim-1,1);
    
  for (int j=1; j<W1_dim; j++) {
    W1(j,j) = (1-LW1.block(j,0,1,j).array().square()).sqrt().prod();  
    for (int i=j+1; i<W1_dim; i++) {
      W1(i,j) = W1(j,j)*LW1(i,j);
    }
  }


  W1 = W1.triangularView<Lower>() * W1.transpose();
  W1.array() = W1_scale * W1.array();

  if (P>0) {

    Block<MatrixXA> W2 = W.bottomRightCorner(P,P);
    
    W2_log_diag = par.segment(ind,W2_dim);
    ind += W2_dim;
    W2.diagonal() = W2_log_diag.array().exp().matrix();
    
    if (nfact_W2 > 0) {
      LW2.setZero();
      for (int j=0; j<nfact_W2; j++) {
	LW2.block(j,j,W2_dim-j,1) = par.segment(ind,W2_dim-j);
	ind += W2_dim - j;
	LW2(j,j) = exp(LW2(j,j));
      }
      W2.template selfadjointView<Eigen::Lower>().rankUpdate(LW2);
    }
  }

   // Rcout << "V1:\n";
   // Rcout << V1 << "\n\n";
   // Rcout << "V2:\n";
   // Rcout << V2 << "\n\n";
  // Rcout << "W:\n";
  // Rcout << W << "\n\n";
  //  print_mat(V1);

}


AScalar ads::eval_f(const Eigen::Ref<VectorXA>& P) {
  
  using Eigen::Lower;
  using std::cout;
  using std::endl;
  using Eigen::Map;

  unwrap_params(P);

  const AScalar  LL = eval_LL();
  AScalar hyperprior;
  if (add_prior) {
    hyperprior = eval_hyperprior();
  } else {
    hyperprior = 0.;
  }

  const AScalar f = LL  + hyperprior;    
  assert(my_finite(f));
  return(f);
}




AScalar ads::eval_LL()
{ 
  // Compute P(Y), including full recursion

  M2t = M20;
  C2t = C20;
  LDLT<MatrixXA> chol_Qt;
  OmegaT = Omega0;
  AScalar log_det_Qt = 0, nuT = nu0;

  for (int t=0; t<T; t++) {
    // run recursion

    set_Gt(A[t], AjIsZero[t], c, u, A_scale, delta, Gt);    
    a2t = Gt.triangularView<Upper>() * M2t;
  
    if (include_H) {
      set_Ht(AjIsZero[t], E[t], phi, M2t, C2t, OmegaT, nuT, delta, Ht);
      // assume bottom P rows of Ht  are all zero
      a2t.middleRows(1,J).array() +=  Ht.array();
    }
 
    Yft = -F1F2[t] * a2t;
    Yft += Ybar[t];
  
    R2t = Gt.triangularView<Upper>() * C2t * Gt.triangularView<Upper>().transpose();
    R2t += W.selfadjointView<Lower>();
    R1t = F2[t] * R2t * F2[t].transpose(); 
    R1t += V2.selfadjointView<Lower>();      
    Qt = F1[t] * R1t * F1[t].transpose();
    Qt +=  V1.selfadjointView<Lower>();
  
    chol_Qt.compute(Qt); // Cholesky of Qt
    assert(chol_Qt.isPositive());

    log_det_Qt += chol_Qt.vectorD().array().log().sum();
  
    S2t = R2t * F1F2[t].transpose();
    QYf = chol_Qt.solve(Yft);
    M2t = S2t * QYf;
    M2t += a2t;
    C2t = -S2t*(chol_Qt.solve(S2t.transpose()));
    C2t += R2t;

    // accumulate terms for Matrix T
    OmegaT += Yft.transpose() * QYf;
    nuT += N;
    assert(my_finite(log_det_Qt));
  }

  LDLT<MatrixXA> chol_DX(OmegaT);
  assert(chol_DX.isPositive());
  AScalar log_det_DX = chol_DX.vectorD().array().log().sum();
  AScalar log_PY = log_const - J*log_det_Qt/2. - nuT*log_det_DX/2.;     
  assert(my_finite(log_PY));
  return(log_PY);
}

AScalar ads::eval_hyperprior() {
  
  // Prior on theta_12
  // K x J matrix normal, diagonal (sparse) covariance matrices

  AScalar prior_theta12 = 0;
  if (include_X) {
    prior_theta12 = MatNorm_logpdf(theta12, mean_theta12,
				   chol_cov_row_theta12,
				   chol_cov_col_theta12,
				   false);
  } 

  // Prior on V1, V2 diag and factors
  // log of diagonal elements (includes Jacobian)


  AScalar prior_diag_V1 = 0;
  for (size_t i=0; i<V1_dim; i++) {
    prior_diag_V1 += halfT_logpdf(V1(i,i), diag_df_V1, diag_scale_V1);
    prior_diag_V1 += V1_log_diag(i); // Jacobian (check this)
  }

  AScalar prior_diag_V2 = 0;
  for (size_t i=0; i<V2_dim; i++) {
    prior_diag_V2 += halfT_logpdf(V2(i,i), diag_df_V2, diag_scale_V2);
    prior_diag_V2 += V2_log_diag(i); // Jacobian (check this)
  }
  
  AScalar prior_fact_V1 = 0;
  if (nfact_V1 > 0) {
    for (int j=0; j < nfact_V1; j++) {
      prior_fact_V1 += halfT_logpdf(LV1(j,j), fact_df_V1, fact_scale_V1);
      prior_fact_V1 += log(LV1(j,j)); // Jacobian (check this)
      for (int i=j+1; i<V1_dim; i++) {
	prior_fact_V1 += T_logpdf(LV1(i,j), fact_df_V1, fact_scale_V1);
      }
    }
  }

  AScalar prior_fact_V2 = 0;
  if (nfact_V2 > 0) {
    for (int j=0; j < nfact_V2; j++) {
      prior_fact_V2 += halfT_logpdf(LV2(j,j), fact_df_V2, fact_scale_V2);
      prior_fact_V2 += log(LV2(j,j)); // Jacobian (check this)
      for (int i=j+1; i<V1_dim; i++) {
	prior_fact_V2 += T_logpdf(LV2(i,j), fact_df_V2, fact_scale_V2);
      }
    }
  }

  AScalar prior_scale_W1 = halfT_logpdf(W1_scale, df_scale_W1, s_scale_W1);
  prior_scale_W1 += log(W1_scale); // Jacobian

  // LKJ prior, including Jacobian (from unwrap_params)
  AScalar prior_corr_W1 = corr_W1_const + (W1_eta-1)*logdet_W1_corr + log_W1_jac;

  AScalar prior_diag_W2 = 0;
  AScalar prior_fact_W2 = 0;

  if (P>0) {

    for (size_t i=0; i<W2_dim; i++) {
      prior_diag_W2 += halfT_logpdf(exp(W2_log_diag(i)), diag_df_W2, diag_scale_W2);
      prior_diag_W2 += W2_log_diag(i); // Jacobian (check this)
    }
    
    if (nfact_W2 > 0) {
      for (int j=0; j < nfact_W2; j++) {
	prior_fact_W2 += halfT_logpdf(LW2(j,j), fact_df_W2, fact_scale_W2);
	prior_fact_W2 += log(LW2(j,j)); // Jacobian (check this)
	for (int i=j+1; i<V1_dim; i++) {
	  prior_fact_W2 += T_logpdf(LW2(i,j), fact_df_W2, fact_scale_W2);
	}
      }
    }
  }
       

  AScalar prior_mats = prior_diag_V1 + prior_fact_V1;
  prior_mats += prior_diag_V2 + prior_fact_V2;
  prior_mats += prior_scale_W1 + prior_corr_W1;

  if (P>0) {
    prior_mats += prior_diag_W2 + prior_fact_W2;
  }
  
  // Priors on c and u

  const AScalar prior_c_mean = norm_logpdf(c_mean, c_mean_pmean, c_mean_psd);
  AScalar prior_c_log_sd = norm_logpdf_trunc0(c_sd, c_sd_pmean, c_sd_psd);
  prior_c_log_sd += c_log_sd; // Jacobian
  const AScalar prior_c_off = -J*LOG_2PI_2 - 0.5*c_off.squaredNorm(); // N(0,1)
  const AScalar prior_c = prior_c_mean + prior_c_log_sd + prior_c_off;

  const AScalar prior_u_mean = norm_logpdf(u_mean, u_mean_pmean, u_mean_psd);
  AScalar prior_u_log_sd = norm_logpdf_trunc0(u_sd, u_sd_pmean, u_sd_psd);
  prior_u_log_sd += u_log_sd; // Jacobian  
 
  const AScalar prior_u_off = -J*LOG_2PI_2 - 0.5*u_off.squaredNorm(); // N(0,1)
  const AScalar prior_u = prior_u_mean + prior_u_log_sd + prior_u_off;

  // Prior on phi
  // J x J matrix normal, diagonal (sparse) covariance matrices

  AScalar prior_phi = 0;
  if (include_H) {
    prior_phi = MatNorm_logpdf(phi, mean_phi,
			       chol_cov_row_phi,
			       chol_cov_col_phi,
			       false);
  }
  
 
  AScalar prior_logit_delta = logit_beta_logpdf(logit_delta, delta_a, delta_b);

  AScalar res= prior_c + prior_u + prior_logit_delta + prior_theta12 + prior_phi + prior_mats;
  assert(my_finite(res));  
 
  return(res);

}


AScalar ads::eval_LL(const Eigen::Ref<VectorXA>& P){
  
  unwrap_params(P);
  AScalar LL = eval_LL();
  
  return(LL);
 
}

AScalar ads::eval_hyperprior(const Eigen::Ref<VectorXA>& P) {
  
  unwrap_params(P);
  AScalar hyperprior = eval_hyperprior();
  
  return(hyperprior);
 
}





#endif

