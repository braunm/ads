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
  std::vector<SparseMatrixXA> F2; // each is N(1+P) x (1+Jb+P), row major
  std::vector<SparseMatrixXA> F1F2; // each is N x (1+Jb+P), row major
  std::vector<VectorXA> A; // national advertising, Jb
  std::vector<MatrixXA> Ybar; // each is  N x J
  std::vector<VectorXA> AjIsZero; // 1 if A_j == 0, 0 otherwise
  std::vector<VectorXA> E; // number of new creatives added for each brand

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

  AScalar delta_a;
  AScalar delta_b;

  AScalar diag_scale_V;
  AScalar diag_mode_V;
  AScalar fact_scale_V;
  AScalar fact_mode_V;

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

  int J; // number of brands
  int Jb; // number of brands that advertise
  int N; // number of cities
  int T; // number of weeks
  int K; // covariates with stationary parameters
  int P; // covariates with nonstationary parameters
  int nfact_V; // factors to estimate V
  int nfact_W1; // factors to estimate W2 (if active)
  int nfact_W2; // factors to estimate W2 (if active)

  int V_dim, W_dim, W1_dim, W2_dim, W1_dim_ch2;

  // parameters

  MatrixXA theta12;
  AScalar logit_delta;
  AScalar delta;


  MatrixXA phi; // Jb x J
  VectorXA V_log_diag;
  AScalar W1_scale;
  VectorXA W1_log_diag;
  AScalar W2_scale;
  VectorXA W2_log_diag;
  MatrixXA V; 
  MatrixXA W;  
  MatrixXA LV; 
  MatrixXA LW1;
  MatrixXA LW2;
  AScalar log_W1_jac;

 
  // intermediate values

  MatrixXA Gt;
  MatrixXA Ht;

  MatrixXA a2t;
  MatrixXA Yft;
  MatrixXA Qt; 
  MatrixXA R2t; 
  MatrixXA M2t;
  MatrixXA C2t;
  MatrixXA S2t; 
  MatrixXA OmegaT;
  AScalar nuT;

  AScalar log_const;
  MatrixXA QYf;
  MatrixXA tmpNJ;
  MatrixXA chol_DX_L;
  VectorXA chol_DX_D;
  


  AScalar log_mvgamma_prior;
  AScalar log_mvgamma_post;
  
  // flags for model specification
  bool include_phi;
  bool add_prior;
  bool include_X;
  bool fix_V;
  bool fix_W;
  bool W1_LKJ;

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


  include_phi = as<bool>(flags["include.phi"]);
  add_prior = as<bool>(flags["add.prior"]);
  include_X = as<bool>(flags["include.X"]);


  A_scale = as<double>(flags["A.scale"]);
  fix_V = as<bool>(flags["fix.V"]);
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

  List Elist;
  if (include_phi) {
    Elist = as<List>(data["E"]);
    E.resize(T);
  }


  V_dim = N;
  W1_dim = 1+Jb;
  W2_dim = P;
  W_dim = W1_dim + W2_dim;
  W1_dim_ch2 = W1_dim*(W1_dim-1)/2;

  // number of factors for estimating covariance matrices


  if (!fix_V) {
    nfact_V = as<int>(dimensions["nfact.V"]);
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

    if (include_phi) {
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

  // Reserve V and W, and their factors
  V.resize(V_dim, V_dim);
  W.resize(W_dim, W_dim);
  
  if (!fix_V) {
    LV.resize(V_dim, nfact_V);
    V_log_diag.resize(V_dim);
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

  if (include_phi) {
    phi.resize(Jb,J);
  }


  Rcout << "Allocating memory for intermediate parameters\n";
  
  Gt.resize(1+Jb+P,1+Jb+P);
  M2t.resize(1+Jb+P,J);
  C2t.resize(1+Jb+P,1+Jb+P);
  a2t.resize(1+Jb+P,J);
  Yft.resize(N,J);
  Qt.resize(N,N); 
  R2t.resize(1+Jb+P, 1+Jb+P);
  S2t.resize(1+Jb+P, N);
  OmegaT.resize(J,J);
  QYf.resize(N,J);
  tmpNJ.resize(N,J);


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
    if (include_phi) {
      
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

    if (fix_V || fix_W) {
      Rcout << "Loading any fixed covariance matrices\n";
      fixed_cov = as<const List>(pars["fixed.cov"]);
    }

    if (fix_V) {

      Rcout << "V is fixed\n";
      const Map<MatrixXd> V_d(as<Map<MatrixXd> >(fixed_cov["V"]));
      V = V_d.cast<AScalar>();
      
    } else {

      Rcout << "V is estimated\n";
      const List priors_V = as<List>(priors["V"]); 
      diag_scale_V = as<double>(priors_V["diag.scale"]);
      diag_mode_V = as<double>(priors_V["diag.mode"]);
      fact_scale_V = as<double>(priors_V["fact.scale"]);
      fact_mode_V = as<double>(priors_V["fact.mode"]);
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
  
  if (include_phi) {
    phi = MatrixXA::Map(par.derived().data() + ind, Jb, J);
    ind += Jb*J;
  }

  logit_delta = par(ind++);
  delta = invlogit(logit_delta); 

   // unwrap elements of V and W, which are 
  // modeled as LL + S.  First, the log of diag(S)
  // is unwrapped. Then, if nfact>0, the unique elements of the 
  // factors, by column.  Diagonal elements are 
  // exponentiated for identification.


  if (!fix_V) {
    
    V.setZero();
    V_log_diag = par.segment(ind, V_dim);
    ind += V_dim;
    V.diagonal() = V_log_diag.array().exp().matrix();
    
    if (nfact_V > 0) {
      LV.setZero();
      for (int j=0; j<nfact_V; j++) {
	LV.block(j, j, V_dim - j,1) = par.segment(ind, V_dim - j);
	ind += V_dim - j;
	LV(j, j) = exp(LV(j, j));      
      }
      V.template selfadjointView<Lower>().rankUpdate(LV);      
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
}

AScalar ads::eval_LL()
  { 
    // Compute P(Y), including full recursion
    
    M2t = M20;
    C2t = C20;

    MatrixXA chol_Qt_L = MatrixXA::Identity(N,N);
    VectorXA chol_Qt_D = VectorXA::Zero(N);
 

    OmegaT = Omega0;
    AScalar log_det_Qt = 0;
    nuT = nu0;
    
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

    Qt = F1F2[t] * R2t * F1F2[t].transpose();
    Qt +=  V.selfadjointView<Lower>();
    LDLT(Qt, chol_Qt_L, chol_Qt_D);  
    log_det_Qt += chol_Qt_D.array().log().sum();    


    S2t = R2t * F1F2[t].transpose();

    QYf = chol_Qt_L.triangularView<Lower>().solve(Yft);
    tmpNJ = chol_Qt_D.asDiagonal().inverse() * QYf;
    QYf = chol_Qt_L.transpose().triangularView<Upper>().solve(tmpNJ);    
    M2t = S2t * QYf;
    M2t += a2t;

    MatrixXA tmp1 = chol_Qt_L.triangularView<Lower>().solve(S2t.transpose());
    MatrixXA tmp2 = chol_Qt_D.asDiagonal().inverse() * tmp1;
    C2t = -tmp1.transpose() * tmp2;        
    C2t += R2t;

    // accumulate terms for Matrix T
    OmegaT += Yft.transpose() * QYf;
    nuT += N;
  }


  MatrixXA chol_DX_L = MatrixXA::Identity(J,J);
  VectorXA chol_DX_D = VectorXA::Zero(J);
  LDLT(OmegaT, chol_DX_L, chol_DX_D);
  AScalar log_det_DX = chol_DX_D.array().log().sum();
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


  // Prior on phi
  // J x J matrix normal, diagonal (sparse) covariance matrices
  
  AScalar prior_phi = 0;
  if (include_phi) {
    prior_phi = MatNorm_logpdf(phi, mean_phi,
			       chol_cov_row_phi,
			       chol_cov_col_phi,
			       false);
  }
  

  // Prior on V diag and factors
  // log of diagonal elements (includes Jacobian)

  AScalar prior_diag_V = 0;
  AScalar prior_fact_V = 0;

  if (!fix_V) {
  
    for (size_t i=0; i<V_dim; i++) {
      prior_diag_V += dnormTrunc0_log(V(i,i), diag_mode_V, diag_scale_V);      
      prior_diag_V += V_log_diag(i); // Jacobian
    }
    
    if (nfact_V > 0) {
      for (int j=0; j < nfact_V; j++) {
	prior_fact_V += dnormTrunc0_log(LV(j,j), fact_mode_V, fact_scale_V);
	prior_fact_V += log(LV(j,j)); // Jacobian (check this)
	for (int i=j+1; i<V_dim; i++) {
	  prior_fact_V += dnorm_log(LV(i,j), fact_mode_V, fact_scale_V);	
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
    
  AScalar prior_W2 = prior_diag_W2 + prior_fact_W2;
  AScalar prior_mats = prior_diag_V + prior_fact_V + prior_W1 + prior_W2;
  AScalar prior_logit_delta = dlogitbeta_log(logit_delta, delta_a, delta_b);
  AScalar res =  prior_logit_delta + prior_theta12 + prior_phi + prior_mats;
  return(res);
}

// Afunc
AScalar ads::Afunc(const AScalar& aT, const AScalar& s) {
  AScalar res = log( 1.0 + aT/s);
  return(res);
}

// set_Gt
void ads::set_Gt(const int& tt) {

  Gt.setZero();
  Gt(0,0) = 1.0 - delta;
  for (int j=0; j<Jb; j++) {
    Gt(0, j+1) = Afunc(A[tt](j), A_scale);
    Gt(j+1, j+1) = 1.0;
  }
  if (P>0) {
    Gt.bottomRightCorner(P,P).setIdentity();
  }
}  // end set_Gt


// set_Ht
void ads::set_Ht(const int& tt) {
  Ht.setZero();
  if (include_phi) {
    Ht = E[tt].asDiagonal() * phi; // H2t
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


List ads::par_check(const Eigen::Ref<VectorXA>& P) {

  using Rcpp::Named;
  using Rcpp::wrap;
  using Rcpp::as;

  unwrap_params(P);
  eval_LL();
  // Return values for A


  double Ldelta = CppAD::Value(logit_delta);
  NumericMatrix MV(V.rows(), V.cols());
  NumericMatrix MW(W.rows(), W.cols());
  NumericMatrix M2treturn(M2t.rows(), M2t.cols());
  NumericMatrix C2treturn(C2t.rows(), C2t.cols());
  NumericMatrix OmegaTreturn(OmegaT.rows(), OmegaT.cols());
  NumericMatrix LW1return(LW1.rows(), LW1.cols());

  for (size_t i=0; i<V.rows(); i++) {
    for (size_t j=0; j<V.cols(); j++) {
      MV(i,j) = Value(V(i,j));
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


  List res = List::create(Named("logit_delta") = wrap(Ldelta),
			  Named("V") = wrap(MV),
			  Named("W") = wrap(MW),
			  Named("chol_W1") = wrap(LW1return),
			  Named("M2t") = wrap(M2treturn),
			  Named("C2t") = wrap(C2treturn),
			  Named("OmegaT") = wrap(OmegaTreturn)
			  );
  return(res);
			  			  
}

#endif
