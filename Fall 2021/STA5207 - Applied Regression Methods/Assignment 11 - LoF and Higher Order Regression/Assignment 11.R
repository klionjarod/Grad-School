#Problem 1 (Lack of Fit)
plastic <- read.table("./Assignment 11/plastic.txt", 
                      col.names = c("y","x"), 
                      quote = "\"", comment.char = "")
hardness = plastic$y
time = plastic$x
library(olsrr)
ols_pure_error_anova(lm(hardness~time))


#Problem 2 (Quadratic Model)
solution <- read.table("./Assignment 11/solution.txt", 
                       col.names = c("y","x"), 
                       quote = "\"", comment.char = "")
sol_X = solution$x - mean(solution$x) #center the predictor
sol_Xsq = sol_X^2 #square the predictor
quad = lm(y~sol_X+sol_Xsq, data=solution) #quadratic model
anova(quad)
summary.lm(quad)
linear = lm(y~sol_X, data=solution)
anova(linear, quad) #get the test-statistic


#Problem 3 (Interactions)
properties <- read.table("./Assignment 11/properties.txt",
                         quote = "\"", comment.char = "", header = TRUE)
prop_X1 = properties$x1 - mean(properties$x1) 
prop_X2 = properties$x2 - mean(properties$x2)
prop_X4 = properties$x4 - mean(properties$x4)
#Test 3-way interactions
full = lm(properties$y ~ prop_X1 * prop_X2 * prop_X4)
anova(full)
prop_X1X2 = prop_X1 * prop_X2 
prop_X1X4 = prop_X1 * prop_X4
prop_X2X4 = prop_X2 * prop_X4
model1=lm(properties$y ~ prop_X1 + prop_X2 + prop_X4
          + prop_X1X2 + prop_X1X4 + prop_X2X4)
anova(model1)
anova(model1, full)

