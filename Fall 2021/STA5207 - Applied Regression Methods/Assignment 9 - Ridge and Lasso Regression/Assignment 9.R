#Ridge Regression
library(lmridge)
Assignment <- read.table("./Assignment 9/Assignment.txt", col.names = c("y","x1","x2","x3"))
ridge = lmridge(y~.,data=Assignment, K=seq(0,0.05,0.001))
summary(ridge)
vif(ridge)
plot(ridge)
#First value of lambda where all VIF's are less than 10 is 0.015
lmridge(y~., data=Assignment, K=seq(0.012,0.017,0.001))$coef
#Find the fitted model on the chosen stable value (lambda = 0.07)
ridge2 = lmridge(y~., data=Assignment, K=0.015)
#Get the values for the fitted equation
summary(ridge2)

#Adaptive Lasso Regression
library(msgps)
library(faraway)
alasso=msgps(X=as.matrix(prostate[,-9]), y = prostate$lpsa,
             penalty="alasso", lambda = 0, gamma = 1)
summary(alasso)
#Plot solution paths for the standardized coefficients
plot.msgps(alasso, xvar="t",criterion="bic")
