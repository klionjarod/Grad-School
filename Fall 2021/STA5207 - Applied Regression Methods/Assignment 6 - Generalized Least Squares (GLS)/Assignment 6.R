#QUESTION 1
#-------------------------------------
library(faraway)
#OLS model
model <- lm(divorce ~ unemployed + femlab + marriage + birth + military, data = divusa)
summary.lm(model)

#plot the residuals
plot(divusa$year, model$residuals, main="Plot of Residuals by Year",
     xlab="Year", ylab="Residuals")
abline(h=0)

#Durbin-Watson test
library(car)
durbinWatsonTest(model)

#AR(1) model
library(nlme)
model1=gls(divorce~unemployed+femlab + marriage + birth + military,
           correlation=corAR1(form=~1),method="ML",data=divusa)
summary(model1)

#Question 2
#--------------------------------
library(olsrr)
labor <- read.table("E:/Users/Jarod/Stuff/School Stuff/Grad School/Fall 2021/STA5207/Assignment 6/labor.txt",
                    col.names = c('y', 'x1', 'x2', 'x3', 'x4', 'x5'))
model2=lm(y ~ x1 + x2 + x3 + x4 + x5, data=labor)
ols_plot_resid_stud_fit(model2)
summary.lm(model2)

#perform WLS estimates using 1/xi as weights
model3 = lm(y ~ x1 + x2 + x3 + x4 + x5, data=labor, weights = 1/predict(model2))
ols_plot_resid_stud_fit(model3)
