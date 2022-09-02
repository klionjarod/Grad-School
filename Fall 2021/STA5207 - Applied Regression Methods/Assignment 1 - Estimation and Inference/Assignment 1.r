properties <- read.table("E:/Users/Jarod/Stuff/School Stuff/Grad School/Fall 2021/STA5207/Assignment 1/properties.txt",
                        col.names = c("Rental Rates", "Age", "Expenses", "Vacancy Rates", "Cost"))
#plot(properties)
#store regression results
one=lm(Rental.Rates~Age+Expenses+Vacancy.Rates+Cost,data=properties)
#display regression results
summary.lm(one)

#Create 95% confidence intervals for each beta
confint(one, level=0.95)

#mean rental rate of 5 year old properties with 4.1 tax rate, 0.16 vacancy rate, and cost of $100,000 square feet
x = data.frame(Age=5, Expenses=4.1, Vacancy.Rates=0.16, Cost=100000)
#create 95% confidence interval for the mean response based on these predictor values
predict(one,x,interval=c("confidence"),level=.95,type=c("response"))

#create 95% prediction interval for a response based on these predictor values
predict(one,x,interval=c("prediction"),level=.95,type=c("response"))

#using age as the only predictor
two = lm(Rental.Rates~Age, data = properties)
summary.lm(two)
#using expenses as the only predictor
three = lm(Rental.Rates~Expenses, data = properties)
summary.lm(three)
#using vacancy rate as the only predictor
four = lm(Rental.Rates~Vacancy.Rates, data = properties)
summary.lm(four)
#using cost as the only predictor
five = lm(Rental.Rates~Cost, data = properties)
summary.lm(five)
