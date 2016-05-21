#ifndef __ads
#define __ads

#include <mb_base.h>
#include <except.h>
#include <utilfuncs.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <cppad_atomics.h>
#include <mat_normAD.h>
#include <MVN_AD.h>
#include <LKJ_AD.h>
#include <LDLT_cppad.h>


using Eigen::MatrixBase;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Lower;
using Eigen::Upper;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Rcpp::Rcout;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::IntegerVector;
using Rcpp::as;

typedef CppAD::AD<double> AScalar;

class ads {

  typedef CppAD::AD<double> AScalar;
  typedef Eigen::Matrix<AScalar, Dynamic, Dynamic> MatrixXA;
  typedef Eigen::Matrix<AScalar, Dynamic, 1> VectorXA;
  typedef Eigen::SparseMatrix<AScalar, Eigen::RowMajor> SparseMatrixXA; // why row major?
  typedef Eigen::MappedSparseMatrix<double> MappedSparseXd;

 public:
  
  ads(const List&);

  AScalar eval_f(const Eigen::Ref<VectorXA>&);
  AScalar eval_LL(const Eigen::Ref<VectorXA>&);
  AScalar eval_hyperprior(const Eigen::Ref<VectorXA>&);
  List par_check(const Eigen::Ref<VectorXA>&);
  
  List get_recursion(const Eigen::Ref<VectorXA>&);
  AScalar eval_LL(const bool); // for storing recursion only
  
private:

  template<typename Tpars>
    void unwrap_params(const MatrixBase<Tpars>&);

  AScalar eval_hyperprior();
  void set_Gt(const int&);
  void set_Ht(const int&);
  AScalar Afunc(const AScalar&, const AScalar&);
  AScalar get_log_PA(const int&);
  
  template<typename TC, typename TX>
    void get_cr_mix(const MatrixBase<TC>&,
		    const MatrixBase<TX>&);

 
  // data
  std::vector<MatrixXA> Y; // each is  N x J
  std::vector<MatrixXA> X; // each is N x K
  std::vector<SparseMatrixXA> F1; // each is N x N(1+P), row major
  std::vector<SparseMatrixXA> F2; // each is N(1+P) x (1+Jb+P), row major
  std::vector<SparseMatrixXA> F1F2; // each is N x (1+Jb+P), row major
  std::vector<VectorXA> A; // national advertising, Jb
  std::vector<MatrixXA> Ybar; // each is  N x J
  std::vector<VectorXA> AjIsZero; // 1 if A_j == 0, 0 otherwise
  std::vector<MatrixXA> CM; // Jb x R creative measures

  // for storing recursion matrices
  std::vector<MatrixXA> M2all; // each is  1+P+Jb x J
  std::vector<MatrixXA> C2all; // each is  1+P+Jb x 1+P+Jb 

  
  // priors

  MatrixXA M20; // 1+P+J x J either prior or parameter
  MatrixXA C20; // 1+P+J x 1+P+J
  MatrixXA Omega0;
  AScalar nu0;

  MatrixXA mean_theta12;
  MatrixXA chol_cov_row_theta12;
  MatrixXA chol_cov_col_theta12;
  
  MatrixXA mean_phi;
  MatrixXA chol_cov_row_phi;
  MatrixXA chol_cov_col_phi;
  VectorXA sd_phi;

  AScalar mean_mean_phi;
  AScalar sd_mean_phi;
  AScalar mode_var_phi;
  AScalar scale_var_phi;

  AScalar delta_a;
  AScalar delta_b;

  AScalar diag_scale_V1;
  AScalar diag_mode_V1;
  AScalar fact_scale_V1;
  AScalar fact_mode_V1;

  AScalar diag_scale_V2;
  AScalar diag_mode_V2;
  AScalar fact_scale_V2;
  AScalar fact_mode_V2;

  AScalar mode_scale_W1;
  AScalar s_scale_W1;
  AScalar W1_eta;

  // W1 factor structure
  AScalar diag_scale_W1;
  AScalar diag_mode_W1;
  AScalar fact_scale_W1;
  AScalar fact_mode_W1;

  AScalar diag_scale_W2;
  AScalar diag_mode_W2;
  AScalar fact_scale_W2;
  AScalar fact_mode_W2;


  MatrixXA mean_G1;
  MatrixXA chol_row_G1;
  MatrixXA chol_col_G1;
  MatrixXA mean_G2;
  MatrixXA chol_row_G2;
  MatrixXA chol_col_G2;
  VectorXA mean_G3;
  MatrixXA chol_cov_G3;

  MatrixXA mean_H1;
  MatrixXA chol_row_H1;
  MatrixXA chol_col_H1;

  

  
  VectorXA mean_cr; // R-1 elements
  MatrixXA chol_cov_cr; // R-1 x R-1



  int J; // number of brands
  int Jb; // number of brands that advertise
  int N; // number of cities
  int T; // number of weeks
  int K; // covariates with stationary parameters
  int P; // covariates with nonstationary parameters
  int R; // covariates in creative metric
  int nfact_V1; // factors to estimate V1
  int nfact_V2; // factors to estimate V2
  int nfact_W1; // factors to estimate W2 (if active)
  int nfact_W2; // factors to estimate W2 (if active)

  int V1_dim, V2_dim, W_dim, W1_dim, W2_dim, W1_dim_ch2;

  // parameters

  MatrixXA theta12;
  AScalar logit_delta;
  AScalar delta;


  MatrixXA phi; // Jb x J
  VectorXA V1_log_diag;
  VectorXA V2_log_diag;
  AScalar W1_scale;
  VectorXA W1_log_diag;
  AScalar W2_scale;
  VectorXA W2_log_diag;
  MatrixXA V1;
  MatrixXA V2;
  MatrixXA W;  
  MatrixXA LV1;
  MatrixXA LV2; 
  MatrixXA LW1;
  MatrixXA LW2;
  AScalar log_W1_jac;

  AScalar phi_mean;
  AScalar phi_log_var;
  VectorXA phi_z;
  
  AScalar phi_var;
  AScalar phi_log_sd;
  AScalar phi_sd;

  // For endogeneity of ad spend
  MatrixXA G1; // for logit pi
  MatrixXA G2; // for mean conditional ad spend
  VectorXA G3; // for scale conditional ad spend

  // For endogeneity of ad spend
  MatrixXA H1; // for logit pi

  VectorXA cr;
  VectorXA cr0;

 
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
  AScalar nuT;


  VectorXA logit_PrA0; // logit prob A==0
  VectorXA mean_A;
  VectorXA scale_A;
  VectorXA m2t1;
  VectorXA log_PA;
  VectorXA PA_r;
  VectorXA PA_a;
  MatrixXA prior_G3;

  VectorXA logit_PrE0; // logit prob E==0
  MatrixXA m2tq;    // q component of M2t
    VectorXA log_PE;
    
  VectorXA crMet; // creative metrics

  
  AScalar log_const;
  MatrixXA QYf;
  MatrixXA tmpNJ;
  MatrixXA chol_DX_L;
  VectorXA chol_DX_D;
  MatrixXA chol_Qt_L;
  VectorXA chol_Qt_D;
  MatrixXA tmp1;
  MatrixXA tmp2;
  
  AScalar log_mvgamma_prior;
  AScalar log_mvgamma_post;
  
  // flags for model specification
  bool full_phi;
  bool phi_re;
  bool add_prior;
  bool include_X;
  bool fix_V1;
  bool fix_V2;
  bool fix_W;
  bool W1_LKJ;
  bool use_cr_pars;
  bool endog_A;
  bool endog_E;
  
  AScalar A_scale;
    
}; // end class definition


ads::ads(const List& params)
{
  // Constructor.  Loading in the data

  const List & pars = static_cast<const List&>(const_cast<List &>(params));

  const List data = as<const List>(pars["data"]);
  const List priors = as<const List>(pars["priors"]);
  const List dimensions = as<const List>(pars["dimensions"]);
  const List flags = as<const List>(pars["flags"]);

  T = as<int>(dimensions["T"]);
  N = as<int>(dimensions["N"]);
  J = as<int>(dimensions["J"]);
  Jb = as<int>(dimensions["Jb"]);
  K = as<int>(dimensions["K"]);
  P = as<int>(dimensions["P"]);
  R = as<int>(dimensions["R"]);

  full_phi = as<bool>(flags["full.phi"]);
  phi_re = as<bool>(flags["phi.re"]);
  add_prior = as<bool>(flags["add.prior"]);

  include_X = as<bool>(flags["include.X"]);
  use_cr_pars = as<bool>(flags["use.cr.pars"]);
  endog_A = as<bool>(flags["endog.A"]);
  endog_E = as<bool>(flags["endog.E"]);
  A_scale = as<double>(flags["A.scale"]);
  fix_V1 = as<bool>(flags["fix.V1"]);
  fix_V2 = as<bool>(flags["fix.V2"]);
  fix_W = as<bool>(flags["fix.W"]);
  W1_LKJ = as<bool>(flags["W1.LKJ"]);

  Rcout << "Constructing data\n";

  
  List Xlist;
  if (include_X) {
    Xlist = as<List>(data["X"]);
    X.resize(T);
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

  const List CMlist = as<List>(data["CM"]);
  CM.resize(T);

  M2all.resize(T);
  C2all.resize(T);

  V1_dim = N;
  V2_dim = N*(1+P);
  W1_dim = 1+Jb;
  W2_dim = P;
  W_dim = W1_dim + W2_dim;
  W1_dim_ch2 = W1_dim*(W1_dim-1)/2;

  // number of factors for estimating covariance matrices
  if (!fix_V1) {
    nfact_V1 = as<int>(dimensions["nfact.V1"]);
  }

  if (!fix_V2) {
    nfact_V2 = as<int>(dimensions["nfact.V2"]);
  }
  
  if (!fix_W) {
    if (P>0) {
      nfact_W1 = as<int>(dimensions["nfact.W1"]);
      nfact_W2 = as<int>(dimensions["nfact.W2"]);
    } else {
      nfact_W1 = 0;
      nfact_W2 = 0;
    }
  }

  // start modeling at week 2
  // to use lags for A and CM
  for (int i=0; i<T; i++) {

    
    check_interrupt();
    
    if (include_X) {
      const Map<MatrixXd> Xd(as<Map<MatrixXd> >(Xlist[i]));
      X[i] = Xd.cast<AScalar>();
    }      

    const Map<MatrixXd> Yd(as<Map<MatrixXd> >(Ylist[i]));
    Y[i] = Yd.cast<AScalar>();

    const Map<VectorXd> Ad(as<Map<VectorXd> >(Alist[i]));
    A[i] = Ad.cast<AScalar>();

    AjIsZero[i].resize(Jb);
    for (int j=0; j<Jb; j++) {
      AjIsZero[i](j) = A[i](j)==0 ? 1. : 0.;
    }

    const Map<MatrixXd> CMd(as<Map<MatrixXd> >(CMlist[i]));
    CM[i] = CMd.cast<AScalar>();

    if (!use_cr_pars) {
      if (CM[i].cols() != 1) {
	Rcout << "Warning: use_cr_pars is FALSE. ";
	Rcout << "CM should be 1 column.\n";
      }
    }
  
    const MappedSparseXd F1d(as<MappedSparseXd >(F1list[i]));
    F1[i] = F1d.cast<AScalar>().transpose(); // transpose should force row major

    const MappedSparseXd F2d(as<MappedSparseXd >(F2list[i]));
    F2[i] = F2d.cast<AScalar>().transpose(); // transpose should force row major

    F1[i].makeCompressed();
    F2[i].makeCompressed();
    F1F2[i] = F1[i] * F2[i];

    Ybar[i].resize(N,J);

    M2all[i].resize(1+P+Jb,J);
    C2all[i].resize(1+P+Jb,1+P+Jb);
  }

  // Reserve space for parameters
  if (include_X) {
    theta12.resize(K,J);
  }

  // Reserve V and W, and their factors
  V1.resize(V1_dim, V1_dim);
  V2.resize(V2_dim, V2_dim);
  W.resize(W_dim, W_dim);
  
  if (!fix_V1) {
    LV1.resize(V1_dim, nfact_V1);
    V1_log_diag.resize(V1_dim);
  }

  if (!fix_V2) {
    LV2.resize(V2_dim, nfact_V2);
    V2_log_diag.resize(V2_dim);
  }
  
  if (!fix_W) {        
    LW1.resize(W1_dim, W1_dim);
    W1_log_diag.resize(W1_dim);
    if (P>0) {
      LW2.resize(W2_dim, W2_dim);
      W2_log_diag.resize(W2_dim);
    }    
  }  
  
  Ht = MatrixXA::Zero(Jb,J); // ignoring zeros in bottom P rows

  phi = MatrixXA::Zero(Jb,J);
  phi_z = VectorXA::Zero(Jb);

  if (use_cr_pars) {
    cr0.resize(R-1);
    cr.resize(R);
    cr[0] = 1; // first element is always 1
  }


  Rcout << "Allocating memory for intermediate parameters\n";
  
  Gt.resize(1+Jb+P,1+Jb+P);
  M2t.resize(1+Jb+P,J);
  C2t.resize(1+Jb+P,1+Jb+P);
  a2t.resize(1+Jb+P,J);
  Yft.resize(N,J);
  Qt.resize(N,N);
  R1t.resize(N*(1+P),N*(1+P));
  R2t.resize(1+Jb+P, 1+Jb+P);
  S2t.resize(1+Jb+P, N);
  OmegaT.resize(J,J);
  QYf.resize(N,J);
  tmpNJ.resize(N,J);
  crMet.resize(Jb);
  chol_DX_L.resize(J,J);
  chol_DX_D.resize(J);
  chol_Qt_L.resize(N,N);
  chol_Qt_D.resize(N);
  tmp1.resize(N, 1+Jb+P);
  tmp2.resize(N, 1+Jb+P);

  // endogeneity
  // for A
  G1.resize(Jb,J+1);
  G2.resize(Jb,J+1);
  G3.resize(Jb);
  logit_PrA0.resize(Jb);
  mean_A.resize(Jb);
  scale_A.resize(Jb);
  m2t1.resize(J+1);
  m2t1[0] = 1; // intercept
  PA_r.resize(Jb);
  PA_a.resize(Jb);
  prior_G3.resize(1,1);
  log_PA.resize(T);
  // for E
  m2tq.resize(Jb,1+J);
  for (int j=0; j<Jb; j++) m2tq(j,1) = 1; // intercept
  logit_PrE0.resize(Jb);
  log_PE.resize(T);

  
  Rcout << "Required prior parameters\n";
  
  // These priors are required

  const Map<MatrixXd> M20_d(as<Map<MatrixXd> >(priors["M20"]));
  M20 = M20_d.cast<AScalar>();
  const Map<MatrixXd> C20_d(as<Map<MatrixXd> >(priors["C20"]));
  C20 = C20_d.cast<AScalar>();
  const Map<MatrixXd> Omega_d(as<Map<MatrixXd> >(priors["Omega0"]));
  Omega0 = Omega_d.cast<AScalar>();
  nu0 = as<double>(priors["nu0"]);
  log_mvgamma_prior = log_MVgamma(nu0 / 2.0, J);
  log_mvgamma_post = log_MVgamma((nu0+T*N) / 2.0, J);
  Eigen::LDLT<MatrixXA> chol_Omega0(Omega0);
  const AScalar log_det_Omega0 = chol_Omega0.vectorD().array().log().sum();
  log_const = -(N*T*J) * M_LN_SQRT_PI + log_mvgamma_prior;
  log_const -=  log_mvgamma_post + nu0 * log_det_Omega0 / 2.0;

  Rcout << "Optional prior parameters\n";
  
  // The following priors are optional
  if (add_prior) {
    const List priors_phi = as<List>(priors["phi"]);
    if (full_phi) {
      const Map<MatrixXd> mean_phi_d(as<Map<MatrixXd> >(priors_phi["mean"]));
      mean_phi = mean_phi_d.cast<AScalar>();
      const Map<MatrixXd> chol_cov_row_phi_d(as<Map<MatrixXd> >(priors_phi["chol.row"]));
      chol_cov_row_phi = chol_cov_row_phi_d.cast<AScalar>();
      const Map<MatrixXd> chol_cov_col_phi_d(as<Map<MatrixXd> >(priors_phi["chol.col"]));
      chol_cov_col_phi = chol_cov_col_phi_d.cast<AScalar>();
      
    } else {

      if (phi_re) {
	mean_mean_phi=as<double>(priors_phi["mean.mean"]);
	sd_mean_phi=as<double>(priors_phi["sd.mean"]);
	mode_var_phi=as<double>(priors_phi["mode.var"]);
	scale_var_phi=as<double>(priors_phi["scale.var"]);	
      } else {	
	const Map<MatrixXd> mean_phi_d(as<Map<MatrixXd> >(priors_phi["mean"]));
	mean_phi = mean_phi_d.cast<AScalar>();
	const Map<MatrixXd> chol_cov_col_phi_d(as<Map<MatrixXd> >(priors_phi["chol.col"]));
	chol_cov_col_phi = chol_cov_col_phi_d.cast<AScalar>();
      }      
    }

    Rcout << "Constructor: Prior delta\n";    
    const List priors_delta = as<List>(priors["delta"]);
    delta_a = as<double>(priors_delta["a"]);
    delta_b = as<double>(priors_delta["b"]);
    
    if (include_X) {

      Rcout << "priors for theta12\n";
      
      const List priors_theta12 = as<List>(priors["theta12"]);
      
      const Map<MatrixXd> mean_theta12_d(as<Map<MatrixXd> >(priors_theta12["mean"]));
      mean_theta12 = mean_theta12_d.cast<AScalar>();
      
      const Map<MatrixXd> chol_cov_row_theta12_d(as<Map<MatrixXd> >(priors_theta12["chol.row"]));
      chol_cov_row_theta12 = chol_cov_row_theta12_d.cast<AScalar>();
      
      const Map<MatrixXd> chol_cov_col_theta12_d(as<Map<MatrixXd> >(priors_theta12["chol.col"]));
      chol_cov_col_theta12 = chol_cov_col_theta12_d.cast<AScalar>();
    }


    List fixed_cov;

    if (fix_V1 || fix_V2 || fix_W) {
      Rcout << "Loading any fixed covariance matrices\n";
      fixed_cov = as<const List>(pars["fixed.cov"]);
    }

    if (fix_V1) {
      Rcout << "V1 is fixed\n";
      const Map<MatrixXd> V1_d(as<Map<MatrixXd> >(fixed_cov["V1"]));
      V1 = V1_d.cast<AScalar>();      
    } else {
      Rcout << "V1 is estimated\n";
      const List priors_V1 = as<List>(priors["V1"]); 
      diag_scale_V1 = as<double>(priors_V1["diag.scale"]);
      diag_mode_V1 = as<double>(priors_V1["diag.mode"]);
      fact_scale_V1 = as<double>(priors_V1["fact.scale"]);
      fact_mode_V1 = as<double>(priors_V1["fact.mode"]);
    }


    if (fix_V2) {
      Rcout << "V2 is fixed\n";
      const Map<MatrixXd> V2_d(as<Map<MatrixXd> >(fixed_cov["V2"]));
      V2 = V2_d.cast<AScalar>();      
    } else {
      Rcout << "V2 is estimated\n";
      const List priors_V2 = as<List>(priors["V2"]); 
      diag_scale_V2 = as<double>(priors_V2["diag.scale"]);
      diag_mode_V2 = as<double>(priors_V2["diag.mode"]);
      fact_scale_V2 = as<double>(priors_V2["fact.scale"]);
      fact_mode_V2 = as<double>(priors_V2["fact.mode"]);
    }
    
    
    if (fix_W) {

      Rcout << "W is fixed\n";
      const Map<MatrixXd> W_d(as<Map<MatrixXd> >(fixed_cov["W"]));
      W = W_d.cast<AScalar>();      
      
    } else {
      Rcout << "W is estimated\n";

      const List priors_W1 = as<List>(priors["W1"]);

      if (W1_LKJ) {      
	mode_scale_W1 = as<double>(priors_W1["scale.mode"]);
	s_scale_W1 = as<double>(priors_W1["scale.s"]);
	const double eta = as<double>(priors_W1["eta"]);
	W1_eta = eta;
      } else {
	diag_scale_W1 = as<double>(priors_W1["diag.scale"]);
	diag_mode_W1 = as<double>(priors_W1["diag.mode"]);
	fact_scale_W1 = as<double>(priors_W1["fact.scale"]);
	fact_mode_W1 = as<double>(priors_W1["fact.mode"]);
      }
            
      if (P>0) {
	Rcout << "W2 priors\n";
	const List priors_W2 = as<List>(priors["W2"]); 
	diag_scale_W2 = as<double>(priors_W2["diag.scale"]);
	diag_mode_W2 = as<double>(priors_W2["diag.mode"]);
	fact_scale_W2 = as<double>(priors_W2["fact.scale"]);
	fact_mode_W2 = as<double>(priors_W2["fact.mode"]);
      }
    }


    
    if (endog_A) {

      Rcout << "priors for endogenous A\n";
      
      const List priors_endog_A = as<List>(priors["endog.A"]);
      const Map<MatrixXd> mean_G1_d(as<Map<MatrixXd> >(priors_endog_A["mean.G1"]));
      mean_G1 = mean_G1_d.cast<AScalar>();
      const Map<MatrixXd> chol_row_G1_d(as<Map<MatrixXd> >(priors_endog_A["chol.row.G1"]));
      chol_row_G1 = chol_row_G1_d.cast<AScalar>();      
      const Map<MatrixXd> chol_col_G1_d(as<Map<MatrixXd> >(priors_endog_A["chol.col.G1"]));
      chol_col_G1 = chol_col_G1_d.cast<AScalar>();
   
      const Map<MatrixXd> mean_G2_d(as<Map<MatrixXd> >(priors_endog_A["mean.G2"]));
      mean_G2 = mean_G2_d.cast<AScalar>();
      const Map<MatrixXd> chol_row_G2_d(as<Map<MatrixXd> >(priors_endog_A["chol.row.G2"]));
      chol_row_G2 = chol_row_G2_d.cast<AScalar>();      
      const Map<MatrixXd> chol_col_G2_d(as<Map<MatrixXd> >(priors_endog_A["chol.col.G2"]));
      chol_col_G2 = chol_col_G2_d.cast<AScalar>();
   
      const Map<VectorXd> mean_G3_d(as<Map<VectorXd> >(priors_endog_A["mean.G3"]));
      mean_G3 = mean_G3_d.cast<AScalar>();
      const Map<MatrixXd> chol_cov_G3_d(as<Map<MatrixXd> >(priors_endog_A["chol.cov.G3"]));
      chol_cov_G3 = chol_cov_G3_d.cast<AScalar>();
      
    }
      
      if(endog_E) {
          
          const List priors_endog_E = as<List>(priors["endog.E"]);
          const Map<MatrixXd> mean_H1_d(as<Map<MatrixXd> >(priors_endog_E["mean.H1"]));
          mean_H1 = mean_H1_d.cast<AScalar>();
          const Map<MatrixXd> chol_row_H1_d(as<Map<MatrixXd> >(priors_endog_E["chol.row.H1"]));
          chol_row_H1 = chol_row_H1_d.cast<AScalar>();
          const Map<MatrixXd> chol_col_H1_d(as<Map<MatrixXd> >(priors_endog_E["chol.col.H1"]));
          chol_col_H1 = chol_col_H1_d.cast<AScalar>();

      }
    

    if (use_cr_pars) {
      Rcout << "Creative priors\n";
      const List priors_cr = as<List>(priors["creatives"]);
      
      const Map<VectorXd> mean_cr_d(as<Map<VectorXd> >(priors_cr["mean"]));
      mean_cr = mean_cr_d.cast<AScalar>();
      
      const Map<MatrixXd> chol_cov_cr_d(as<Map<MatrixXd> >(priors_cr["chol.cov"]));
      chol_cov_cr = chol_cov_cr_d.cast<AScalar>();
    }

    
  } // end priors

  
  Rcout << "End constructor\n";
  
}

template<typename Tpars>
void ads::unwrap_params(const MatrixBase<Tpars>& par)
{
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
  
  if (full_phi) {
    phi = MatrixXA::Map(par.derived().data() + ind, Jb, J);
    ind += Jb*J;
  } else {
    phi = MatrixXA::Zero(Jb,J);
    phi_z = par.segment(ind,Jb);
    ind += Jb;
    if (phi_re) {
      phi_mean = par(ind++);
      phi_log_var = par(ind++);
      phi_var = exp(phi_log_var);
      phi_log_sd = phi_log_var * 0.5;
      phi_sd = exp(phi_log_sd);     
      phi.leftCols(Jb) = (phi_z.array() * phi_sd + phi_mean).matrix().asDiagonal();
    } else {
      phi.leftCols(Jb) = phi_z.asDiagonal();
    }
  }

  logit_delta = par(ind++);
  delta = invlogit(logit_delta); 

   // unwrap elements of V and W, which are 
  // modeled as LL + S.  First, the log of diag(S)
  // is unwrapped. Then, if nfact>0, the unique elements of the 
  // factors, by column.  Diagonal elements are 
  // exponentiated for identification.


  if (!fix_V1) {
    
    // Build V1
    V1.setZero();
    V1_log_diag = par.segment(ind, V1_dim);
    ind += V1_dim;
    V1.diagonal() = V1_log_diag.array().exp().matrix();
    
    if (nfact_V1 > 0) {
      LV1.setZero();
      for (int j=0; j<nfact_V1; j++) {
	LV1.block(j, j, V1_dim - j,1) = par.segment(ind, V1_dim - j);
	ind += V1_dim - j;
	LV1(j, j) = exp(LV1(j, j));      
      }
      V1.template selfadjointView<Lower>().rankUpdate(LV1);
      //V1 += LV1 * LV1.transpose();      
    }
  }

    // Build V2

  if (!fix_V2) { 
    V2.setZero();
    V2_log_diag = par.segment(ind, V2_dim);
    ind += V2_dim;
    V2.diagonal() = V2_log_diag.array().exp().matrix();
    
    if (nfact_V2 > 0) {
      LV2.setZero();
      for (int j=0; j<nfact_V2; j++) {
	LV2.block(j, j, V2_dim - j,1) = par.segment(ind, V2_dim - j);
	ind += V2_dim - j;
	LV2(j, j) = exp(LV2(j, j));      
      }
      V2.template selfadjointView<Lower>().rankUpdate(LV2);      
    }    
  }

  if (!fix_W) {
  
    // W is a scaled correlation matrix.
    // The log scale factor is the first
    // element.  Then there is a transform of the lower
    // triangle of the correlation matrix.
    
    W.setZero();

    Eigen::Block<MatrixXA> W1 = W.topLeftCorner(1+Jb,1+Jb);
 
    if (W1_LKJ) {
      LW1.setZero();   
      W1_scale = exp(par(ind++));
      // transform to lower Cholesky and get Jacobian
      log_W1_jac = lkj_unwrap(par.segment(ind, W1_dim_ch2), LW1);
      ind += W1_dim_ch2;
      W1 = LW1 * LW1.transpose();
      W1.array() = W1_scale * W1.array();
    } else {
      W1_log_diag = par.segment(ind,W1_dim);
      ind += W1_dim;
      W1.diagonal() = W1_log_diag.array().exp().matrix();
      
      if (nfact_W1 > 0) {
	LW1.setZero();
	for (int j=0; j<nfact_W1; j++) {
	  LW1.block(j,j,W1_dim-j,1) = par.segment(ind,W1_dim-j);
	  ind += W1_dim - j;
	  LW1(j,j) = exp(LW1(j,j));
	}
	W1.template selfadjointView<Lower>().rankUpdate(LW1);
      }
    }
    
    // work on W2 now
    if (P>0) {
      
      Eigen::Block<MatrixXA> W2 = W.bottomRightCorner(P,P);
      
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
	W2.template selfadjointView<Lower>().rankUpdate(LW2);
      }
    }
  }

  // unwrap endogeneity parameters

  if (endog_A) {
    G1 = MatrixXA::Map(par.derived().data() + ind, Jb, J+1);
    ind += Jb*J;
    G2 = MatrixXA::Map(par.derived().data() + ind, Jb, J+1);    
    ind += Jb*J;
    G3 = VectorXA::Map(par.derived().data() + ind, Jb);
    ind += Jb;
  }

    if (endog_E) {
        H1 = MatrixXA::Map(par.derived().data() + ind, Jb, J+1);
        ind += Jb*J;
    }


  

  // parameters for creatives
  if (use_cr_pars) {
    cr0 = par.segment(ind, R-1);
    cr.tail(R-1) = cr0;
    ind += R-1;
  }
}

AScalar ads::eval_LL(const bool store=false)
  { 
    // Compute P(Y), including full recursion

    //    Rcout << "store = " << store << "\n";
    
    M2t = M20;
    C2t = C20;

    OmegaT = Omega0;
    AScalar log_det_Qt = 0;
    nuT = nu0;
    log_PA.setZero();
    log_PE.setZero();

    // start modeling at week 2
    // to use lags for A and CM    
  for (int t=0; t<T; t++) {
    check_interrupt();

    // run recursion
    
    set_Gt(t);
    a2t = Gt.triangularView<Upper>() * M2t;
    set_Ht(t);
    // assume bottom P rows of Ht  are all zero
    a2t.middleRows(1,Jb).array() +=  Ht.array();
    
    Yft = -F1F2[t] * a2t;
    Yft += Ybar[t];
  
    R2t = Gt.triangularView<Upper>() * C2t * Gt.triangularView<Upper>().transpose();
    R2t += W.selfadjointView<Lower>();

    R1t = F2[t] * R2t * F2[t].transpose();
    R1t += V2.selfadjointView<Lower>();
    
    Qt = F1[t] * R1t * F1[t].transpose();
    Qt +=  V1.selfadjointView<Lower>();  

    LDLT(Qt, chol_Qt_L, chol_Qt_D);  
    log_det_Qt += chol_Qt_D.array().log().sum();    

    S2t = R2t * F1F2[t].transpose();
    QYf = chol_Qt_L.triangularView<Lower>().solve(Yft);
    tmpNJ = chol_Qt_D.asDiagonal().inverse() * QYf;

    if (endog_A) {
      m2t1.tail(J) = M2t.row(0).transpose(); // col vector with intercept
      logit_PrA0 = G1 * m2t1;
      mean_A = (G2 * m2t1).array().exp().matrix();
      scale_A = G3.array().exp().matrix();
      log_PA(t) = get_log_PA(t);
    }
      
    if (endog_E) {
        for(int j=0; j < Jb; j++) {
              m2tq.row(j).tail(J) = M2t.row(1+j); // col vector for each of the Jb rows
              logit_PrE0 = H1.col(j) * m2tq;
            if( CM[t](j) == 0 ) log_PE(t) += log(invlogit(logit_PrE0(j))); else log_PE(t) += log(1-invlogit(logit_PrE0(j)));
          }
    }
    
    
    QYf = chol_Qt_L.transpose().triangularView<Upper>().solve(tmpNJ);    
    M2t = S2t * QYf;
    M2t += a2t;

    
    tmp1 = chol_Qt_L.triangularView<Lower>().solve(S2t.transpose());
    tmp2 = chol_Qt_D.asDiagonal().inverse() * tmp1;
    C2t = -tmp1.transpose() * tmp2;        
    C2t += R2t;

    if (store) {
      M2all[t] = M2t;
      C2all[t] = C2t;
    }
    

    // accumulate terms for Matrix T
    OmegaT += Yft.transpose() * QYf;
    nuT += N;    
  }

  LDLT(OmegaT, chol_DX_L, chol_DX_D);
  AScalar log_det_DX = chol_DX_D.array().log().sum();
  AScalar log_PY = log_const - J*log_det_Qt/2. - nuT*log_det_DX/2.;     
  AScalar res = log_PY + log_PA.sum() + log_PE.sum();

  return(res);
}

AScalar ads::get_log_PA(const int& tt) {

  PA_a.array() = mean_A.array()/scale_A.array();
  PA_r.array() = PA_a.array() * mean_A.array();
  VectorXA logres(Jb);
  AScalar PrA0, log_fA, Att, res;


  for (int j=0; j<Jb; j++) {
    PrA0 = invlogit(logit_PrA0(j));
    Att = A[tt](j)/A_scale;
    if (Att==0) {
      logres(j) = log(PrA0);
    } else {
      log_fA = dgamma_log(Att,PA_r(j),PA_a(j));    
      logres(j) = log(1-PrA0) + log_fA;
    }
  }  
  res = logres.sum();
  assert(my_finite(res));
  return(res);
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


  // Prior on phi
  // J x J matrix normal, diagonal (sparse) covariance matrices
  
  AScalar prior_phi = 0;
  if (full_phi) {
    prior_phi = MatNorm_logpdf(phi, mean_phi,
			       chol_cov_row_phi,
			       chol_cov_col_phi,
			       false);
  } else {

    if (phi_re) {
      for (int i=0; i<Jb; i++) {
	prior_phi += dnorm_log(phi_z(i),0,1);
      }
      prior_phi += dnorm_log(phi_mean, mean_mean_phi, sd_mean_phi);
      prior_phi += dnormTrunc0_log(phi_var, mode_var_phi, scale_var_phi);
      prior_phi += phi_log_var; // Inv Jacobian for log_var -> var
    } else {
      MatrixXA pp(1,1);
      MVN_logpdf(phi_z, mean_phi.col(0), chol_cov_row_phi, pp, false);
      prior_phi += pp(0,0);
    }
  }
  

  // Prior on V diag and factors
  // log of diagonal elements (includes Jacobian)

  AScalar prior_diag_V1 = 0;
  AScalar prior_fact_V1 = 0;
  if (!fix_V1) {
  
    for (size_t i=0; i<V1_dim; i++) {
      prior_diag_V1 += dnormTrunc0_log(exp(V1_log_diag(i)),
				      diag_mode_V1, diag_scale_V1);      
      prior_diag_V1 += V1_log_diag(i); // Jacobian
    }
    
    if (nfact_V1 > 0) {
      for (int j=0; j < nfact_V1; j++) {
	prior_fact_V1 += dnormTrunc0_log(LV1(j,j), fact_mode_V1, fact_scale_V1);
	prior_fact_V1 += log(LV1(j,j)); // Jacobian
	for (int i=j+1; i<V1_dim; i++) {
	  prior_fact_V1 += dnorm_log(LV1(i,j), fact_mode_V1, fact_scale_V1);	
	}
      }
    }
  }

  AScalar prior_diag_V2 = 0;
  AScalar prior_fact_V2 = 0;
  if (!fix_V2) {
  
    for (size_t i=0; i<V2_dim; i++) {
      prior_diag_V2 += dnormTrunc0_log(exp(V2_log_diag(i)),
				      diag_mode_V2, diag_scale_V2);      
      prior_diag_V2 += V2_log_diag(i); // Jacobian
    }
    
    if (nfact_V2 > 0) {
      for (int j=0; j < nfact_V2; j++) {
	prior_fact_V2 += dnormTrunc0_log(LV2(j,j), fact_mode_V2, fact_scale_V2);
	prior_fact_V2 += log(LV2(j,j)); // Jacobian
	for (int i=j+1; i<V2_dim; i++) {
	  prior_fact_V2 += dnorm_log(LV2(i,j), fact_mode_V2, fact_scale_V2);	
	}
      }
    }
  }

  AScalar prior_scale_W1 = 0;
  AScalar prior_corr_W1 = 0;
  AScalar prior_W1 = 0;
  AScalar prior_diag_W1 = 0;
  AScalar prior_fact_W1 = 0;
  AScalar prior_diag_W2 = 0;
  AScalar prior_fact_W2 = 0;
  
  if (!fix_W) {

    if (W1_LKJ) {
      prior_scale_W1 = dnormTrunc0_log(W1_scale, mode_scale_W1, s_scale_W1);      
      prior_scale_W1 += log(W1_scale); // Jacobian
      prior_corr_W1 = lkj_chol_logpdf(LW1, W1_eta);
      prior_W1 = prior_scale_W1 + prior_corr_W1 + log_W1_jac;
    } else {
      for (size_t i=0; i<W1_dim; i++) {
	prior_diag_W1 += dnormTrunc0_log(exp(W1_log_diag(i)), diag_mode_W1, diag_scale_W1);	
	prior_diag_W1 += W1_log_diag(i); // Jacobian
      }
      
      if (nfact_W1 > 0) {
	for (int j=0; j < nfact_W1; j++) {
	  prior_fact_W1 += dnormTrunc0_log(LW1(j,j), fact_mode_W1, fact_scale_W1);
          prior_fact_W1 += log(LW1(j,j)); // Jacobian
          for (int i=j+1; i<W1_dim; i++) {
	    prior_fact_W1 += dnorm_log(LW1(i,j), fact_mode_W1, fact_scale_W1);	
	  }
	}
      }
      prior_W1 = prior_diag_W1 + prior_fact_W1;
    }
    
    if (P>0) {
      
      for (size_t i=0; i<W2_dim; i++) {
	prior_diag_W2 += dnormTrunc0_log(exp(W2_log_diag(i)), diag_mode_W2, diag_scale_W2);	
	prior_diag_W2 += W2_log_diag(i); // Jacobian
      }
      
      if (nfact_W2 > 0) {
	for (int j=0; j < nfact_W2; j++) {
	  prior_fact_W2 += dnormTrunc0_log(LW2(j,j), fact_mode_W2, fact_scale_W2);
          prior_fact_W2 += log(LW2(j,j)); // Jacobian
          for (int i=j+1; i<W2_dim; i++) {
	    prior_fact_W2 += dnorm_log(LW2(i,j), fact_mode_W2, fact_scale_W2);	
	  }
	}
      }      
    }
  }

  AScalar prior_endog_A = 0;
  if (endog_A) {    
    AScalar prior_G1 = MatNorm_logpdf(G1, mean_G1, chol_row_G1, chol_col_G1, false);
    AScalar prior_G2 = MatNorm_logpdf(G2, mean_G2, chol_row_G2, chol_col_G2, false);
    MVN_logpdf(G3, mean_G3.col(0), chol_cov_G3, prior_G3, false);
    prior_endog_A = prior_G1 + prior_G2 + prior_G3(0,0);
  }
    
    AScalar prior_endog_E = 0;
    if(endog_E) {
        prior_endog_E = MatNorm_logpdf(H1, mean_H1, chol_row_H1, chol_col_H1, false);
    }
  
 
  AScalar prior_cr = 0.0;
  if (use_cr_pars) {
    MatrixXA crp(1,1);
    MVN_logpdf(cr0, mean_cr, chol_cov_cr, crp, false);
    prior_cr = crp(0,0);
  }
  
  AScalar prior_V = prior_diag_V1 + prior_fact_V1 + prior_diag_V2 + prior_fact_V2;
  AScalar prior_W2 = prior_diag_W2 + prior_fact_W2;
  AScalar prior_mats = prior_V + prior_W1 + prior_W2;
  AScalar prior_logit_delta = dlogitbeta_log(logit_delta, delta_a, delta_b);
  AScalar res =  prior_logit_delta + prior_theta12 +
    prior_phi + prior_mats + prior_cr + prior_endog_A + prior_endog_E;
  
  return(res);
}

// Afunc
AScalar ads::Afunc(const AScalar& aT, const AScalar& s) {
  AScalar res = log1p(aT/s);
  return(res);
}


// set_Gt
void ads::set_Gt(const int& tt) {

  Gt.setZero();
  Gt(0,0) = 1.0 - delta;
  for (int j=0; j<Jb; j++) {
    Gt(0, j+1) = Afunc(A[tt](j), 1.0);
    Gt(j+1, j+1) = 1.0;
  }
  if (P>0) {
    Gt.bottomRightCorner(P,P).setIdentity();
  }
}  // end set_Gt


// creative mix
template<typename TC, typename TX>
  void ads::get_cr_mix(const MatrixBase<TC>& CX,
		       const MatrixBase<TX>& out_) {
  // cr : vector of parameters for creatives
  // CX input matrix
  // creative measure for each brand

  MatrixBase<TX> & out = const_cast<MatrixBase<TX>& >(out_); 
  
  assert(out.size()==Jb);
  assert(CX.rows()==Jb);
  assert(CX.cols()==R);
  assert(cr[0]==1);

  out = CX * cr;
}


// set_Ht
void ads::set_Ht(const int& tt) {

  Ht.setZero();
  if (use_cr_pars) {
    get_cr_mix(CM[tt], crMet);
    Ht = crMet.asDiagonal() * phi; // H2t
  } else {
    Ht = CM[tt].asDiagonal() * phi;
  }
    
} // end set_Ht


// Optional functions to return values outside of the tape

AScalar ads::eval_LL(const Eigen::Ref<VectorXA>& P){
  unwrap_params(P);
  AScalar LL = eval_LL(); 
  return LL;
}

AScalar ads::eval_hyperprior(const Eigen::Ref<VectorXA>& P) {
  unwrap_params(P);
  AScalar hyperprior = eval_hyperprior();
  return hyperprior;
}

AScalar ads::eval_f(const Eigen::Ref<VectorXA>& P) {
  unwrap_params(P);
  AScalar  f = eval_LL();
  AScalar hyperprior;
  if (add_prior) {
    f += eval_hyperprior();
  }
  return f;
}

List ads::get_recursion(const Eigen::Ref<VectorXA>& P) {

  using Rcpp::Named;
  using Rcpp::wrap;
  using Rcpp::as;
  
  unwrap_params(P);
  eval_LL(true); // sets store recursion flag

  Rcpp::List M2return(T);
  Rcpp::List C2return(T);
  for (size_t tt=0; tt<T; tt++) {
    Rcpp::NumericMatrix MM(M2t.rows(), M2t.cols());
    for (size_t col=0; col < M2t.cols(); col++) {
      for (size_t row=0; row < M2t.rows(); row++) {
  	MM(row, col) = Value(M2all[tt](row, col));
      }
    }
    M2return(tt) = wrap(MM);
    
    Rcpp::NumericMatrix CC(C2t.rows(), C2t.cols());
    for (size_t col=0; col < C2t.cols(); col++) {
      for (size_t row=0; row < C2t.rows(); row++) {
  	CC(row, col) = Value(C2all[tt](row, col));
      }
    }
    C2return(tt) = wrap(CC);
  }

  List res = List::create(Named("M2t") = wrap(M2return),
			  Named("C2t") = wrap(C2return)
			  );
  return(res);
}


List ads::par_check(const Eigen::Ref<VectorXA>& P) {

  using Rcpp::Named;
  using Rcpp::wrap;
  using Rcpp::as;

  unwrap_params(P);
  eval_LL();
  // Return values for A


  double Ldelta = CppAD::Value(logit_delta);
  NumericMatrix MV1(V1.rows(), V1.cols());
  NumericMatrix MV2(V2.rows(), V2.cols());
  NumericMatrix MW(W.rows(), W.cols());
  NumericMatrix M2treturn(M2t.rows(), M2t.cols());
  NumericMatrix C2treturn(C2t.rows(), C2t.cols());
  NumericMatrix OmegaTreturn(OmegaT.rows(), OmegaT.cols());
  NumericMatrix LW1return(LW1.rows(), LW1.cols());
  NumericMatrix phiReturn(phi.rows(), phi.cols());
  NumericVector crReturn(cr.size());


  MatrixXA DF1 = F1[1];
  MatrixXA DF2 = F2[1];
  MatrixXA DF1F2 = F1F2[1];

  NumericMatrix MF1(DF1.rows(), DF1.cols());
  NumericMatrix MF2(DF2.rows(), DF2.cols());
  NumericMatrix MF1F2(DF1F2.rows(), DF1F2.cols());

  
  for (size_t i=0; i<DF1.rows(); i++) {
    for (size_t j=0; j<DF1.cols(); j++) {
      MF1(i,j) = Value(DF1(i,j));
    }
  }

  for (size_t i=0; i<DF2.rows(); i++) {
    for (size_t j=0; j<DF2.cols(); j++) {
      MF2(i,j) = Value(DF2(i,j));
    }
  }

  for (size_t i=0; i<DF1F2.rows(); i++) {
    for (size_t j=0; j<DF1F2.cols(); j++) {
      MF1F2(i,j) = Value(DF1F2(i,j));
    }
  }

  
  for (size_t i=0; i<V1.rows(); i++) {
    for (size_t j=0; j<V1.cols(); j++) {
      MV1(i,j) = Value(V1(i,j));
    }
  }

  for (size_t i=0; i<V2.rows(); i++) {
    for (size_t j=0; j<V2.cols(); j++) {
      MV2(i,j) = Value(V2(i,j));
    }
  }


  for (size_t i=0; i<W.rows(); i++) {
    for (size_t j=0; j<W.cols(); j++) {
      MW(i,j) = Value(W(i,j));
    }
  }

  for (size_t i=0; i<LW1.rows(); i++) {
    for (size_t j=0; j<LW1.cols(); j++) {
      LW1return(i,j) = Value(LW1(i,j));
    }
  }

  for (size_t i=0; i<M2t.rows(); i++) {
    for (size_t j=0; j<M2t.cols(); j++) {
      M2treturn(i,j) = Value(M2t(i,j));
    }
  }


  for (size_t i=0; i<C2t.rows(); i++) {
    for (size_t j=0; j<C2t.cols(); j++) {
      C2treturn(i,j) = Value(C2t(i,j));
    }
  }


  for (size_t i=0; i<OmegaT.rows(); i++) {
    for (size_t j=0; j<OmegaT.cols(); j++) {
      OmegaTreturn(i,j) = Value(OmegaT(i,j));
    }
  }

  for (size_t i=0; i<phi.rows(); i++) {
    for (size_t j=0; j<phi.cols(); j++) {
      phiReturn(i,j) = Value(phi(i,j));
    }
  }

  if (use_cr_pars) {
    for (size_t i=0; i<cr.size(); i++) { 
      crReturn(i) = Value(cr(i));
    }
  }

  Rcpp::List Hreturn(T);
  for (size_t tt=0; tt<T; tt++) {
    set_Ht(tt);
    Rcpp::NumericMatrix HH(Ht.rows(), Ht.cols());
    for (size_t col=0; col < Ht.cols(); col++) {
      for (size_t row=0; row < Ht.rows(); row++) {
  	HH(row, col) = Value(Ht(row, col));
      }
    }
    Hreturn(tt) = wrap(HH);
  }


  List res = List::create(Named("logit_delta") = wrap(Ldelta),
			  Named("F1") = wrap(MF1),
			  Named("F2") = wrap(MF2),
			  Named("F1F2") = wrap(MF1F2),	  
			  Named("V1") = wrap(MV1),
			  Named("V2") = wrap(MV2),
			  Named("W") = wrap(MW),
			  Named("chol_W1") = wrap(LW1return),
			  Named("M2t") = wrap(M2treturn),
			  Named("C2t") = wrap(C2treturn),
			  Named("OmegaT") = wrap(OmegaTreturn),
			  Named("phi") = wrap(phiReturn),
			  Named("cr") = wrap(crReturn),
			  Named("Ht") = wrap(Hreturn)
			  );
  return(res);
			  			  
}

#endif
