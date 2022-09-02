library(faraway)
library(nortest)
library(olsrr)
model = lm(gamble~sex+status+income+verbal, data=teengamb)

#PROBLEM 1
#plot residuals by predictors
residuals=residuals(model)
#graphics window stuff
par(mfrow=c(2,2))
plot(teengamb$sex,residuals,main='Sex by residuals', xlab='Sex', ylab='residuals')
plot(teengamb$status,residuals,main='Status by residuals', xlab='Status', ylab='residuals')
plot(teengamb$income,residuals,main='Income by residuals', xlab='Income', ylab='residuals')
plot(teengamb$verbal,residuals,main='Verbal by residuals', xlab='Verbal', ylab='residuals')

#PROBLEM 2
#Perform regression using predicted values and abs val of residuals
summary.lm(lm(abs(residuals)~fitted(model)))

#PROBLEM 3
#Normality tests of the residuals 
ols_test_normality(model)

#histogram
ols_plot_resid_hist(model)
#boxplot
boxplot(residuals,main='Boxplot of Residuals')
#normal prob
ols_plot_resid_qq(model)

#PROBLEM 4
#in terms of response
ols_plot_resid_stud_fit(model)

#PROBLEM 5
#Look for influential observations on one fitted value
ols_plot_dffits(model)

#PROBLEM 6
#in terms of predictors
ols_plot_resid_lev(model)

#PROBLEM 7
#Look for influential observations on all fitted value
ols_plot_cooksd_chart(model)

#PROBLEM 8
#Look for influential observations on a coefficient estimate
ols_plot_dfbetas(model)

#PROBLEM 9
#partial regression plots
ols_plot_added_variable(model)

#PROBLEM 10
#t-tests for each predictor
summary.lm(model)
