#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

//' Fitting Simple Linear Models
//'
//' 'lm_cpp' is used to fit simple linear models.
//'
//' @param y a vector of response variable.
//' @param x a vector of predictor variable.
//'
//' @return coefficients a named vector of coefficients.
//' @return se the standard errors.
//' @return ci the 95% confidence intervals.
//' @return residuals the residuals, that is response minus fitted values.
//' @return fitted.values the fitted mean values.
//'
//' @export
//'
// [[Rcpp::export]]
Rcpp::List lm_cpp(const arma::vec & y, const arma::vec & x) {

  // Check if the length of x and y are the same
  if (x.n_rows != y.n_rows) {
    Rcpp::stop("Lengths of x and y must be the same.");
  }

  // Number of observations
  int n = x.size();

  // Create the design matrix
  arma::mat X(n, 2);
  X.col(0) = arma::ones<arma::vec>(n);
  X.col(1) = x;

  // Compute coefficients
  arma::vec beta = arma::solve(X.t() * X, X.t() * y);

  // Compute the predicted values
  arma::vec yhat = X * beta;

  // Compute residuals
  arma::vec res = y - yhat;

  // Compute standard errors
  double rss = arma::sum(arma::square(res));
  double sigmasq_hat = rss / (n - 2);
  arma::mat CovMtx = sigmasq_hat * arma::inv(X.t() * X);
  arma::vec se = arma::sqrt(arma::diagvec(CovMtx));

  // Compute confidence intervals
  double t = R::qt(0.975, n - 2, true, false);
  arma::mat ci = arma::join_rows(beta - t * se, beta + t * se);

  return Rcpp::List::create(
    Rcpp::Named("coefficients") = beta,
    Rcpp::Named("SE") = se,
    Rcpp::Named("CI") = ci,
    Rcpp::Named("residuals") = res,
    Rcpp::Named("fitted.values") = yhat
  );
}
