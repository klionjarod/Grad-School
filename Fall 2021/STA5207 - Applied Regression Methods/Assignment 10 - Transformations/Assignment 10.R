#Problem 1
#Simple Linear Regression
prb1 <- read.table("./Assignment 10/A10Q1.txt", col.names = c("x","y"))
x = prb1[,1]
y = prb1[,2]
plot(x,y)
model = lm(y~x)
library(olsrr)
ols_plot_resid_fit(model)
ols_plot_resid_qq(model)
library(MASS)
bc1 = boxcox(y~x)
#Find optimal value of lambda
lambda1=bc1$x[which.max(bc1$y)]
yprime1=y^lambda1
plot(x,yprime1)
slr = lm(yprime1~x)
ols_plot_resid_fit(slr)
ols_plot_resid_hist(slr)
ols_plot_resid_qq(slr)

#Problem 2
library(faraway)
gamble2 = teengamb[,5] + 1 #Since some values of gamble are 0
mlr = lm(gamble2~sex + status + income + verbal, data=teengamb)
ols_plot_resid_fit(mlr)
ols_plot_resid_hist(mlr)
ols_plot_resid_qq(mlr)
ols_plot_added_variable(mlr)

bc2 = boxcox(gamble2~sex + status + income + verbal, data=teengamb)
lambda2 = bc2$x[which.max(bc2$y)]
model1 = lm(gamble2^lambda2~sex + status + income + verbal, data=teengamb)
summary.lm(model1)
ols_plot_resid_fit(model1)
ols_plot_resid_hist(model1)
ols_plot_resid_qq(model1)
