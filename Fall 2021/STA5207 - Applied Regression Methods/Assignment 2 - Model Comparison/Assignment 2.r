spider <- read.table("E:/Users/Jarod/Stuff/School Stuff/Grad School/Fall 2021/STA5207/Assignment 2/spider.txt",
                     col.names = c('y', 'x1', 'x2', 'x3', 'x4'))

#QUESTION 1
#using x1 and x4 only
one=lm(y~x1+x4,data=spider)
anova(one)
summary.lm(one)
#adding x2 to the regression
two=lm(y~x1+x2+x4, data=spider)
anova(two)
anova(one,two)

#QUESTION 2
full=lm(y~x1+x2+x3+x4,data=spider)
no_x3=lm(y~x1+x3+x4,data=spider)
anova(full)
anova(no_x3)
anova(no_x3,full)

#QUESTION 3
only_x1 = lm(y~x1, data=spider)
three = lm(y~x1+x3+x4,data=spider)
anova(only_x1)
anova(three)
anova(only_x1, three)

#QUESTION 4
four = lm(y~I(x1+x2)+x3+x4, data=spider)
anova(four)
anova(four,full)
