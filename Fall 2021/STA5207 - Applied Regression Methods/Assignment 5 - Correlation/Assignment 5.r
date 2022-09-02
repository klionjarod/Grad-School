library(faraway)
library(olsrr)

model = lm(Employed ~ GNP.deflator + GNP + Unemployed + Armed.Forces +
             Population + Year, data=longley)
residuals=residuals(model)
summary.lm(model)


#Problem 2
ols_plot_resid_fit(model)
summary.lm(lm(abs(residuals)~fitted(model)))
#standard plots
ols_plot_resid_hist(model)
#boxplot
boxplot(residuals,main='Boxplot of Residuals')
#normal prob
ols_plot_resid_qq(model)

#Problem 4
cor(longley)

#Problem 5/7
ols_coll_diag(model)
round(ols_eigen_cindex(model), 4)
