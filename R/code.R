#' Fitting Simple Linear Models
#'
#' 'lm_r' is used to fit simple linear models.
#'
#' @param y a vector of response variable.
#' @param x a vector of predictor variable.
#'
#' @return coefficients a named vector of coefficients.
#' @return se the standard errors.
#' @return ci the 95% conf idence intervals.
#' @return residuals the residuals, that is response minus fitted values.
#' @return fitted.values the fitted mean values.
#'
#' @export

lm_r = function(y, x){

  if(length(y) != length(x)){
    stop("Lengths of x and y must be the same.")
  }

  n = length(x)
  X = cbind(rep(1, n), x)

  beta = solve(t(X) %*% X, t(X) %*% y)
  yhat = X %*% beta
  res = y - yhat

  rss = sum(res^2)
  sigmasq_hat = rss / (n - 2)
  CovMtx = sigmasq_hat * solve(t(X) %*% X)
  se = sqrt(diag(CovMtx))

  t = qt(0.975, n - 2)
  ci = matrix(c(beta - t * se, beta + t * se), 2, 2)

  return(list("coefficients" = beta, "SE" = se, "CI" = ci,
              "residuals" = res, "fitted.values" = yhat))
}

