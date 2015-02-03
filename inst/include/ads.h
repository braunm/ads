#ifndef __ads
#define __ads

#include <mb_base.h>
#include <except.h>
#include <utilfuncs.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <cppad_atomic/atomics.h>
#include <mat_normAD.h>

using Eigen::MatrixBase;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Lower;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Rcpp::Rcout;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::IntegerVector;
using Rcpp::as;

typedef CppAD::AD<double> AScalar;

AScalar logT(const AScalar& z,
	     const AScalar& v,
	     const AScalar& s) {
  
  AScalar res = lgamma(0.5*(v+1)) - lgamma(0.5*v);
  res -= 0.5 * ((v+1) * log1p(z*z/(s*s*v)) + log(v));
  res -= log(s) + M_LN_SQRT_PI;
  return(res);
}

AScalar logHalfT(const AScalar& z,
	     const AScalar& v,
	     const AScalar& s) {
  
  AScalar res = M_LN2 + logT(z, v, s);
  return(res);
}




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

  
private:

  template<typename Tpars>
    void unwrap_params(const MatrixBase<Tpars>&);

  AScalar eval_LL();
  AScalar eval_hyperprior();
  void set_Gt(const int&);
  void set_Ht(const int&);
  AScalar Afunc(const AScalar&, const AScalar&);
 
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

  // the following for W1 only used if W1_LKJ is off
  AScalar diag_scale_W1;
  AScalar diag_df_W1;
  AScalar fact_scale_W1;
  AScalar fact_df_W1;
    
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
  int nfact_W1; // factors to estimate W1 (if active)
  int nfact_W2; // factors to estimate W2 (if active)

  int V1_dim, V2_dim, W_dim, W1_dim, W2_dim;

  // parameters

  MatrixXA theta12;
  AScalar logit_delta;
  AScalar delta;
  //  AScalar c_mean;
  // AScalar c_log_sd;
  // AScalar c_sd;
  //  VectorXA c_off; // J copy wearout parameters offset
  // AScalar u_mean;
  // AScalar u_log_sd;
  // AScalar u_sd;
  // VectorXA u_off; // J ad wearout parameters offset

  AScalar c_a, c_b, u_a, u_b;

  
  MatrixXA phi; // J x J
  VectorXA V1_log_diag;
  VectorXA V2_log_diag;
  AScalar W1_scale;
  AScalar W2_scale;
  VectorXA W1_log_diag;
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
  AScalar nuT;
  MatrixXA Pneg;

  AScalar log_const;
  MatrixXA QYf;

  VectorXA c;
  VectorXA u;
  VectorXA logit_c;
  VectorXA logit_u;

  AScalar log_mvgamma_prior;
  AScalar log_mvgamma_post;
  
  // flags for model specification
  bool include_H;
  bool add_prior;
  bool include_X;
  bool include_c;
  bool include_u;
  bool W1_LKJ;
  bool fix_V1;
  bool fix_V2;
  bool fix_W;

  AScalar A_scale;
    
}; // end class definition



ads::ads(const List& params)
{
  // Constructor.  Loading in the data

  const List & pars = static_cast<const List&>(const_cast<List &>(params));

  Rcout << "break 1\n";
  
  const List data = as<const List>(pars["data"]);
  const List priors = as<const List>(pars["priors"]);
  const List dimensions = as<const List>(pars["dimensions"]);
  const List flags = as<const List>(pars["flags"]);


  Rcout << "break 2\n";
  
  T = as<int>(dimensions["T"]);
  N = as<int>(dimensions["N"]);
  J = as<int>(dimensions["J"]);
  K = as<int>(dimensions["K"]);
  P = as<int>(dimensions["P"]);

  include_H = as<bool>(flags["include.H"]);
  add_prior = as<bool>(flags["add.prior"]);
  include_X = as<bool>(flags["include.X"]);
  include_c = as<bool>(flags["include.c"]);
  include_u = as<bool>(flags["include.u"]);
  W1_LKJ = as<bool>(flags["W1.LKJ"]);
  A_scale = as<double>(flags["A.scale"]);
  fix_V1 = as<bool>(flags["fix.V1"]);
  fix_V2 = as<bool>(flags["fix.V2"]);
  fix_W = as<bool>(flags["fix.W"]);

  

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

  List Elist;
  if (include_H) {
    Elist = as<List>(data["E"]);
    E.resize(T);
  }


  V1_dim = N;
  V2_dim = N*(P+1);
  W1_dim = 1+J;
  W2_dim = P;
  W_dim = W1_dim + W2_dim;



  
  // number of factors for estimating covariance matrices


  if (!fix_V1) {
    nfact_V1 = as<int>(dimensions["nfact.V1"]);
  }
  if (!fix_V2) {
    nfact_V2 = as<int>(dimensions["nfact.V2"]);
  }

  if (!fix_W) {
    if (P>0) {
      nfact_W2 = as<int>(dimensions["nfact.W2"]);
    } else {
      nfact_W2 = 0;
    }
    
    if (W1_LKJ) {
      nfact_W1 = as<int>(dimensions["nfact.W1"]);
    } else {
      nfact_W1 = 0;
    }
  }
  
 
  
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


  // Reserve space for parameters
  if (include_X) {
    theta12.resize(K,J);
  }
  
  // Reserve V1, V2 and W, and their factors
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
    if (P>0) {
      LW2.resize(W2_dim, W2_dim);
      W2_log_diag.resize(W2_dim);
    }    
    if (!W1_LKJ) {
      LW1.resize(W1_dim,W1_dim);
      W1_log_diag.resize(W1_dim);
    }
  }  
  
  if (include_c) {
    c.resize(J);
    logit_c.resize(J);
   }

  if (include_u) {
    u.resize(J);
    logit_u.resize(J);
  }

  
  if (include_H) {
    phi.resize(J,J);
    Ht = MatrixXA::Zero(J,J); // ignoring zeros in bottom P rows
    Pneg.resize(J,J);   
  }

  Rcout << "Allocating memory for intermediate parameters\n";
  
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
    if (include_H) {
      
      const List priors_phi = as<List>(priors["phi"]);
      const Map<MatrixXd> mean_phi_d(as<Map<MatrixXd> >(priors_phi["mean"]));
      mean_phi = mean_phi_d.cast<AScalar>();
      const Map<MatrixXd> chol_cov_row_phi_d(as<Map<MatrixXd> >(priors_phi["chol.row"]));
      chol_cov_row_phi = chol_cov_row_phi_d.cast<AScalar>();
      const Map<MatrixXd> chol_cov_col_phi_d(as<Map<MatrixXd> >(priors_phi["chol.col"]));
      chol_cov_col_phi = chol_cov_col_phi_d.cast<AScalar>();
    }

    Rcout << "Constructor: Prior delta\n";
    
    const List priors_delta = as<List>(priors["delta"]);
    delta_a = as<double>(priors_delta["a"]);
    delta_b = as<double>(priors_delta["b"]);


    
    if (include_c) {

      Rcout << "priors for c\n";
      
      const List priors_c = as<List>(priors["c"]);
      c_a = as<double>(priors_c["a"]);
      c_b = as<double>(priors_c["b"]);
      
      /* c_mean_pmean = as<double>(priors_c["mean.mean"]); */
      /* c_mean_psd = as<double>(priors_c["mean.sd"]); */
      /* c_sd_pmean = as<double>(priors_c["sd.mean"]); */
      /* c_sd_psd = as<double>(priors_c["sd.sd"]); */
    }


    if (include_u) {

      Rcout << "priors for u\n";
      
      const List priors_u = as<List>(priors["u"]);
      u_a = as<double>(priors_u["a"]);
      u_b = as<double>(priors_u["b"]);
      
      /* u_mean_pmean = as<double>(priors_u["mean.mean"]); */
      /* u_mean_psd = as<double>(priors_u["mean.sd"]); */
      /* u_sd_pmean = as<double>(priors_u["sd.mean"]); */
      /* u_sd_psd = as<double>(priors_u["sd.sd"]); */
    }
    
    
    Rcout << "break 3\n";
    
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
      diag_df_V1 = as<double>(priors_V1["diag.df"]);
      fact_scale_V1 = as<double>(priors_V1["fact.scale"]);
      fact_df_V1 = as<double>(priors_V1["fact.df"]);
    }
    
    if (fix_V2) {

      Rcout << "V2 is fixed\n";
      const Map<MatrixXd> V2_d(as<Map<MatrixXd> >(fixed_cov["V2"]));
      V2 = V2_d.cast<AScalar>();
      
    } else {
      Rcout << "V2 is estimated\n";
      const List priors_V2 = as<List>(priors["V2"]); 
      diag_scale_V2 = as<double>(priors_V2["diag.scale"]);
      diag_df_V2 = as<double>(priors_V2["diag.df"]);
      fact_scale_V2 = as<double>(priors_V2["fact.scale"]);
      fact_df_V2 = as<double>(priors_V2["fact.df"]);
    }
    
    if (fix_W) {

      Rcout << "W is fixed\n";
      const Map<MatrixXd> W_d(as<Map<MatrixXd> >(fixed_cov["W"]));
      W = W_d.cast<AScalar>();      
      
    } else {
      Rcout << "W is estimated\n";
      if (W1_LKJ) {
	const List priors_W1 = as<List>(priors["W1"]);
	df_scale_W1 = as<double>(priors_W1["scale.df"]);
	s_scale_W1 = as<double>(priors_W1["scale.s"]);
	const double eta = as<double>(priors_W1["eta"]);
	W1_eta = eta;
	
	// from LKJ, Eq. 16
	double t1 = 0;
	double t2 = 0;
	for (int i=1; i<=(W1_dim-1); i++) {
	  t1 += (2.0 * eta - 2.0 + W1_dim) * (W1_dim - i);
	  double tmp = eta + 0.5 * (W1_dim - i - 1.0);
	  t2 += (W1_dim-i) * (2.0 * lgamma(tmp) - lgamma(2.0 * tmp));
	}
	corr_W1_const = t1 * M_LN2 + t2;
      } else {
	const List priors_W1 = as<List>(priors["W1"]);
	diag_scale_W1 = as<double>(priors_W1["diag.scale"]);
	diag_df_W1 = as<double>(priors_W1["diag.df"]);
	fact_scale_W1 = as<double>(priors_W1["fact.scale"]);
	fact_df_W1 = as<double>(priors_W1["fact.df"]);
      }
      
      if (P>0) {
	const List priors_W2 = as<List>(priors["W2"]); 
	diag_scale_W2 = as<double>(priors_W2["diag.scale"]);
	diag_df_W2 = as<double>(priors_W2["diag.df"]);
	fact_scale_W2 = as<double>(priors_W2["fact.scale"]);
	fact_df_W2 = as<double>(priors_W2["fact.df"]);
      }
    }
  }

  
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

  // for c and u, the parameter is an offset agains the 
  // mean.  So c_j = c_mean + par[ind+j]
  if (include_c) {

    logit_c = par.segment(ind, J);
    ind += J;
     
    /* c_mean = par(ind++); //ind increments after pull */
    /* c_log_sd = par(ind++); // ind increments after pull */
    /* c_sd = exp(c_log_sd); */
    /* c_off = par.segment(ind,J); // N(0,1) prior */
    /* ind += J; */
    /* logit_c.array() = c_sd * c_off.array() + c_mean; */

    c.array() = logit_c.array().exp()/(1+logit_c.array().exp()); 
   }


    if (include_u) {

    logit_u = par.segment(ind, J);
    ind += J;

    /* u_mean = par(ind++); //ind increments after pull */
    /* u_log_sd = par(ind++); // ind increments after pull */
    /* u_sd = exp(u_log_sd); */
    /* u_off = par.segment(ind,J); // N(0,1) prior */
    /* ind += J; */
    /* logit_u.array() = u_sd * u_off.array() + u_mean; */

    u.array() = logit_u.array().exp()/(1+logit_u.array().exp());
  }


  
  
  if (include_H) {
    phi = MatrixXA::Map(par.derived().data() + ind, J, J);
    ind += J*J;
  }

  logit_delta = par(ind++);
  delta = invlogit(logit_delta); 

   // unwrap elements of V1, V2 and W, which are 
  // modeled as LL + S.  First, the log of diag(S)
  // is unwrapped. Then, if nfact>0, the unique elements of the 
  // factors, by column.  Diagonal elements are 
  // exponentiated for identification.


  if (!fix_V1) {
    
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
      V1.template selfadjointView<Eigen::Lower>().rankUpdate(LV1);
    }
  }

  if (!fix_V2) {
  
    V2_log_diag = par.segment(ind, V2_dim);
    ind += V2_dim;
    V2.setZero();
    V2.diagonal() = V2_log_diag.array().exp().matrix();
    
    if (nfact_V2 > 0) {
      LV2.setZero();
      for (int j=0; j<nfact_V2; j++) {
	LV2.block(j, j, V2_dim-j, 1) = par.segment(ind, V2_dim - j);
	ind += V2_dim - j;
	LV2(j, j) = exp(LV2(j, j));
      }
      V2.template selfadjointView<Eigen::Lower>().rankUpdate(LV2);
    }
  }

  if (!fix_W) {
  
    // W is a scaled correlation matrix.
    // The log scale factor is the first
    // element.  Then there is a transform of the lower
    // triangle of the correlation matrix.
    
    W.setZero();
    
    if (W1_LKJ) {
      W1_scale = exp(par(ind++));
      LW1.setZero();
      // copy terms to lower triangle
      logdet_W1_corr = 0;
      log_W1_jac = 0;
      for (int j=0; j<W1_dim-1; j++) {
	for (int i=j+1; i<W1_dim; i++) {
	  LW1(i,j) = tanh(par(ind++));
	  AScalar tmp = log1p(-pow(LW1(i, j), 2));
	  logdet_W1_corr += tmp;
	  log_W1_jac += 0.5 * (W1_dim-j) * tmp;
	}
      }
      
      Eigen::Block<MatrixXA> W1 = W.topLeftCorner(1+J,1+J);
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
      
    } else {
      
      Eigen::Block<MatrixXA> W1 = W.topLeftCorner(1+J,1+J);
      
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
	W1.template selfadjointView<Eigen::Lower>().rankUpdate(LW1);
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
	W2.template selfadjointView<Eigen::Lower>().rankUpdate(LW2);
      }
    }
  }
}

AScalar ads::eval_LL()
  { 
    // Compute P(Y), including full recursion
    using Eigen::Upper;

    /* for (size_t i=0; i<T; i++) { */
    /*   Rcout << "A[" << i << "] = " << A[i] << "\n"; */
    /* } */
    
    
    M2t = M20;
    C2t = C20;
    Eigen::LDLT<MatrixXA> chol_Qt;
    OmegaT = Omega0;
    AScalar log_det_Qt = 0;
    nuT = nu0;
    
  for (int t=0; t<T; t++) {
    check_interrupt();

    // run recursion
    
    set_Gt(t);

    //  Rcout << "Gt[" << t << "] =\n " << Gt << "\n\n";
    
    a2t = Gt.triangularView<Upper>() * M2t;
  
    if (include_H) {
      set_Ht(t);
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
  }

  Eigen::LDLT<MatrixXA> chol_DX(OmegaT);
  AScalar log_det_DX = chol_DX.vectorD().array().log().sum();
  AScalar log_PY = log_const - J*log_det_Qt/2. - nuT*log_det_DX/2.;     
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
  AScalar prior_fact_V1 = 0;

  if (!fix_V1) {
  
    for (size_t i=0; i<V1_dim; i++) {
      //    prior_diag_V1 += dhalft_log(V1(i,i), diag_df_V1, diag_scale_V1);
      prior_diag_V1 += logHalfT(V1(i,i), diag_df_V1, diag_scale_V1);
      prior_diag_V1 += V1_log_diag(i); // Jacobian (check this)
    }
    
    if (nfact_V1 > 0) {
      for (int j=0; j < nfact_V1; j++) {
	//   prior_fact_V1 += dhalft_log(LV1(j,j), fact_df_V1, fact_scale_V1);
	prior_fact_V1 += logHalfT(LV1(j,j), fact_df_V1, fact_scale_V1);      
	prior_fact_V1 += log(LV1(j,j)); // Jacobian (check this)
	for (int i=j+1; i<V1_dim; i++) {
	//	prior_fact_V1 += dt_log(LV1(i,j), fact_df_V1, fact_scale_V1);
	  prior_fact_V1 += logT(LV1(i,j), fact_df_V1, fact_scale_V1);
	}
      }
    }
  }

  AScalar prior_diag_V2 = 0;
  AScalar prior_fact_V2 = 0;


  if (!fix_V2) {
  
    for (size_t i=0; i<V2_dim; i++) {
      //   prior_diag_V2 += dhalft_log(V2(i,i), diag_df_V2, diag_scale_V2);
      prior_diag_V2 += logHalfT(V2(i,i), diag_df_V2, diag_scale_V2);
      prior_diag_V2 += V2_log_diag(i); // Jacobian (check this)
    }
    
    
    if (nfact_V2 > 0) {
      for (int j=0; j < nfact_V2; j++) {
	//   prior_fact_V2 += dhalft_log(LV2(j,j), fact_df_V2, fact_scale_V2);
	prior_fact_V2 += logHalfT(LV2(j,j), fact_df_V2, fact_scale_V2);
	prior_fact_V2 += log(LV2(j,j)); // Jacobian (check this)
	for (int i=j+1; i<V2_dim; i++) {
	  //   prior_fact_V2 += dt_log(LV2(i,j), fact_df_V2, fact_scale_V2);
	  prior_fact_V2 += logT(LV2(i,j), fact_df_V2, fact_scale_V2);
	}
      }
    }
  }
  
  AScalar prior_scale_W1 = 0;
  AScalar prior_corr_W1 = 0;
  AScalar prior_diag_W1 = 0;
  AScalar prior_fact_W1 = 0;
  AScalar prior_W1 = 0;

  AScalar prior_diag_W2 = 0;
  AScalar prior_fact_W2 = 0;
 
  

  if (!fix_W) {
  
    if(W1_LKJ) {
      // prior_scale_W1 = dhalft_log(W1_scale, df_scale_W1, s_scale_W1);
      prior_scale_W1 = logHalfT(W1_scale, df_scale_W1, s_scale_W1);
      prior_scale_W1 += log(W1_scale); // Jacobian
      
      // LKJ prior, including Jacobian (from unwrap_params)
      prior_corr_W1 = corr_W1_const + (W1_eta-1)*logdet_W1_corr + log_W1_jac;
      
      prior_W1 = prior_scale_W1 + prior_corr_W1;
      
    } else {
      
      // NEED PRIOR ON DIAG_W1!
      
      for (size_t i=0; i<W1_dim; i++) {
	prior_diag_W1 += logHalfT(exp(W1_log_diag(i)), diag_df_W1, diag_scale_W1);
	prior_diag_W1 += W1_log_diag(i); // Jacobian (check this)
      }
      
      if (nfact_W1 > 0) {
	for (int j=0; j < nfact_W1; j++) {
	  //     prior_fact_W1 += dhalft_log(LW1(j,j), fact_df_W1, fact_scale_W1);
	  prior_fact_W1 += logHalfT(LW1(j,j), fact_df_W1, fact_scale_W1);
	  prior_fact_W1 += log(LW1(j,j)); // Jacobian (check this)
	  for (int i=j+1; i<W1_dim; i++) {
	    //    prior_fact_W1 += dt_log(LW1(i,j), fact_df_W1, fact_scale_W1);
	    prior_fact_W1 += logT(LW1(i,j), fact_df_W1, fact_scale_W1);
	  }
	}
      }
      prior_W1 = prior_diag_W1 + prior_fact_W1; 
    }
    
    
    if (P>0) {
      
      for (size_t i=0; i<W2_dim; i++) {
	//     prior_diag_W2 += dhalft_log(exp(W2_log_diag(i)), diag_df_W2, diag_scale_W2);
	prior_diag_W2 += logHalfT(exp(W2_log_diag(i)), diag_df_W2, diag_scale_W2);
	prior_diag_W2 += W2_log_diag(i); // Jacobian (check this)
      }
      
      if (nfact_W2 > 0) {
	for (int j=0; j < nfact_W2; j++) {
	  //    prior_fact_W2 += dhalft_log(LW2(j,j), fact_df_W2, fact_scale_W2);
	  prior_fact_W2 += logHalfT(LW2(j,j), fact_df_W2, fact_scale_W2);
          prior_fact_W2 += log(LW2(j,j)); // Jacobian (check this)
          for (int i=j+1; i<W2_dim; i++) {
	    // prior_fact_W2 += dt_log(LW2(i,j), fact_df_W2, fact_scale_W2);
	    prior_fact_W2 += logT(LW2(i,j), fact_df_W2, fact_scale_W2);
	  }
	}
      }
    }
  }
    
    AScalar prior_W2 = prior_diag_W2 + prior_fact_W2;
    
    AScalar prior_mats = prior_diag_V1 + prior_fact_V1;
    prior_mats += prior_diag_V2 + prior_fact_V2;
    prior_mats += prior_W1 + prior_W2;
    
    
    // Priors on c and u
    
    AScalar prior_u = 0;
    AScalar prior_c = 0;
    
    if (include_c) {
      /* const AScalar prior_c_mean = dnorm_log(c_mean, c_mean_pmean, c_mean_psd); */
      /* AScalar prior_c_log_sd = dnormTrunc0_log(c_sd, c_sd_pmean, c_sd_psd); */
        /* prior_c_log_sd += c_log_sd; // Jacobian */
        /* const AScalar prior_c_off = -J*M_LN_SQRT_2PI - 0.5*c_off.squaredNorm(); // N(0,1) */
        /* const AScalar prior_c = prior_c_mean + prior_c_log_sd + prior_c_off; */

      for (int jj=0; jj<J; jj++) {      
	prior_c += dlogitbeta_log(logit_c(jj), c_a, c_b);
      }      
    }
    
    if (include_u) {

      /* const AScalar prior_u_mean = dnorm_log(u_mean, u_mean_pmean, u_mean_psd); */
      /* AScalar prior_u_log_sd = dnormTrunc0_log(u_sd, u_sd_pmean, u_sd_psd); */
      /* prior_u_log_sd += u_log_sd; // Jacobian */
      
      /* const AScalar prior_u_off = -J*M_LN_SQRT_2PI - 0.5*u_off.squaredNorm(); // N(0,1) */
      /* const AScalar prior_u = prior_u_mean + prior_u_log_sd + prior_u_off; */
      
      for (int jj=0; jj<J; jj++) {      
	prior_u += dlogitbeta_log(logit_u(jj), u_a, u_b);
      }  
    }
    

  // Prior on phi
  // J x J matrix normal, diagonal (sparse) covariance matrices

    AScalar prior_phi = 0;
    if (include_H) {
      prior_phi = MatNorm_logpdf(phi, mean_phi,
				 chol_cov_row_phi,
				 chol_cov_col_phi,
				 false);
    }
    
    AScalar prior_logit_delta = dlogitbeta_log(logit_delta, delta_a, delta_b);
    AScalar res= prior_c + prior_u + prior_logit_delta + prior_theta12 + prior_phi + prior_mats;
    
    return(res);
}

// Afunc
AScalar ads::Afunc(const AScalar& aT, const AScalar& A_scale) {
  AScalar res = log( 1.0 + aT );
  return(res);
}



// set_Gt
void ads::set_Gt(const int& tt) {

  // Logit specification for decay of effectiveness

  
  Gt.setZero();
  Gt(0,0) = 1.0 - delta;
  for (int j=0; j<J; j++) {
    Gt(0, j+1) = Afunc(A[tt](j), A_scale);
    
    if (include_c) {
      if (include_u) {
	// c and u	
	// Gt(j+1, j+1) = 1.0 - c(j) - u(j)*A[tt](j)/A_scale-delta * AjIsZero[tt](j);
	Gt(j+1, j+1) = exp(-(c(j) + u(j)*A[tt](j)/A_scale));
      } else {
	// c, not u
	Gt(j+1, j+1) = exp(-c(j));	
      }
    } else {
      if (include_u) {
	// u, not c
	Gt(j+1, j+1) = exp(-u(j)*A[tt](j)/A_scale);
	//Gt(j+1, j+1) = exp(-u(j)*log1p(A[tt](j)));

      } else {
	// neither c nor u
	Gt(j+1, j+1) = 1.0;
      }
    }
  } // end loop over J

  
  if (P>0) {
    Gt.bottomRightCorner(P,P).setIdentity();
  }
}  // end set_Gt




// set_Ht
void ads::set_Ht(const int& tt) {

  Ht = E[tt].asDiagonal() * phi; // H2t
  Ht.array().colwise() += delta * AjIsZero[tt].array(); //H1t
  
  // Estimate probability that sign(qij)<0
  
  AScalar ct = nuT - P - 2*J;
  Eigen::Matrix<AScalar, Dynamic, Dynamic> Pneg(J,J);
  Pneg.setZero();
  
  if (ct<30) {    
    // use CDF of student T
    for (size_t col=0; col<J; col++) {
      for (size_t row=0; row<J; row++) {
	AScalar mm = M2t(row+1, col);
	AScalar dd = C2t(row+1, row+1) * OmegaT(col, col);
	AScalar IB = incbeta(ct*mm*mm  / (ct*mm*mm + dd*dd), 0.5, 0.5*ct);
	Pneg(row,col) = 0.5 * (1.0 - sign(mm)*IB);
      }
    }   
  } else {
    // use CDF of normal    
    for (size_t col=0; col<J; col++) {
      for (size_t row=0; row<J; row++) {
	AScalar mm = M2t(row+1, col);
	AScalar dd = C2t(row+1, row+1) * OmegaT(col, col);
	Pneg(row, col) = exp(pnorm_log(0, mm, dd/ct));
      }
    }    
  }
  Ht.array() = (1.0 - 2.0 * Pneg.array()) * Ht.array();
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


List ads::par_check(const Eigen::Ref<VectorXA>& P) {

  using Rcpp::Named;
  using Rcpp::wrap;
  using Rcpp::as;

  unwrap_params(P);

  // Return values for A



    
  
  NumericVector LC(logit_c.size());
  NumericVector LU(logit_u.size());
  double Ldelta = CppAD::Value(logit_delta);
  NumericMatrix MV1(V1.rows(), V1.cols());
  NumericMatrix MV2(V2.rows(), V2.cols());
  NumericMatrix MW(W.rows(), W.cols());

  for (size_t i=0; i<logit_c.size(); i++) {
    LC(i) = Value(logit_c(i));
  }
  for (size_t i=0; i<logit_u.size(); i++) {
    LU(i) = Value(logit_u(i));
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

  Rcpp::List Areturn(T);  
  for (size_t ti=0; ti<A.size(); ti++) {
    //   Rcout << "A[" << ti << "] = " << A[ti] << "\n";
    Rcpp::NumericVector Atj(J);
    for (size_t j=0; j<J; j++) {
      Atj(j) = Value(A[ti](j));
    }
    Areturn(ti) = wrap(Atj);
  }

  Rcpp::List Greturn(T);  
  for (size_t tt=0; tt<T; tt++) {
    set_Gt(tt);
    Rcpp::NumericMatrix GG(Gt.rows(), Gt.cols());
    for (size_t col=0; col < Gt.cols(); col++) {
      for (size_t row=0; row < Gt.rows(); row++) {
	GG(row, col) = Value(Gt(row, col));
      }
    }
    Greturn(tt) = wrap(GG);
  }

  List res = List::create(Named("logit_c") = wrap(LC),
			  Named("logit_u") = wrap(LU),
			  Named("logit_delta") = wrap(Ldelta),
			  Named("V1") = wrap(MV1),
			  Named("V2") = wrap(MV2),
			  Named("W") = wrap(MW),
			  Named("A") = wrap(Areturn),
			  Named("Gt") = wrap(Greturn)
			  );
  return(res);
			  			  
}

#endif

