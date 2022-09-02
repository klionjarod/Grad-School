library(olsrr)

#Read in data
assignment <- read.table("./Assignment.txt", col.names = c("y","x1","x2","x3"))

#Check for multicollinearity
ols_coll_diag(model <- lm(y ~ x1 + x2 + x3, data=assignment))

#Standardize the predictors
st_data = scale(assignment[,-1], center=T, scale=T)

#Principal component analysis
#calculate principal directions
prc = prcomp(st_data)
#show variance proportions
summary(prc)
#display principal directions
prc
#obtain all the principal components
pcs = st_data %*% prc$rotation

#principal component regression using only the first 2 PCs
model = lm(assignment$y ~ pcs[,1:2])
summary.lm(model)

#PCR with all 4 principal components
summary.lm(lm(assignment$y~pcs))
           