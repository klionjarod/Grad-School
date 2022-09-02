library(faraway) #For dataset
library(MASS) #For Huber's method
library(quantreg) #For least absolute deviations
library(olsrr)

#ordinary least squares
ols_model = lm(stack.loss ~ Air.Flow + Water.Temp + Acid.Conc., data = stackloss)
summary.lm(ols_model)
opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(ols_model, las = 1)


#Huber's method
huber = rlm(stack.loss ~ Air.Flow + Water.Temp + Acid.Conc., data = stackloss)
summary(huber)

#Least absolute deviations 
lad = rq(stack.loss ~ Air.Flow + Water.Temp + Acid.Conc., data = stackloss)
summary(lad)

#Remove last observation and re-do
stackloss2 = head(stackloss,-1)
summary.lm(lm(stack.loss ~ Air.Flow + Water.Temp + Acid.Conc., data = stackloss2))
summary(rlm(stack.loss ~ Air.Flow + Water.Temp + Acid.Conc., data = stackloss2))
summary(rq(stack.loss ~ Air.Flow + Water.Temp + Acid.Conc., data = stackloss2))