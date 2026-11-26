//! Spatial regression (spreg): OLS with diagnostics, with shared linear algebra for later
//! maximum-likelihood and GM estimators.

pub mod brent;
pub mod dist;
pub mod gm_common;
pub mod gm_error;
pub mod gm_lag;
pub mod linalg;
pub mod ml_common;
pub mod ml_error;
pub mod ml_lag;
pub mod ols;
pub mod panel;
pub mod panel_error_re;
pub mod panel_gm_kkp;
pub mod sur_error_ml;
pub mod tsls;
