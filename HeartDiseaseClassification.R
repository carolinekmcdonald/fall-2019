#############ST443 Final Project###########
#################Final Version#################

####loading packages####
library(ISLR)
library(leaps)
library(boot)
library(readr)
library(stargazer)
library(randomForest)

#clearing environment
rm(list=ls())

#setting working directory
setwd("~/Desktop/LSE/ST443/Final Project")

###loading dataset###
ht <- read_csv("framingham.csv")

###removing all rows with NAs from dataset###
ht=data.frame(na.omit(ht))

#Note, only 15% of the original dataset has heart disease.
#Thus a random split runs the risk of having an uneven distribution of the
#classification variable (heart disease).
sum(ht$TenYearCHD)/3656

##creating predict() method for regsubsets()
predict_regsubsets = function(object, newdata, id, ...) {
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coef_i = coef(object, id = id)
  mat[, names(coef_i)] %*% coef_i
}

#set seed to generate indices for training data
set.seed(1234)

# Generate training (75% of data) and testing data sets (25%) 
train_ind = sample(seq_len(nrow(ht)), size = floor(0.75*nrow(ht)))

# ID for training data
train=(ht[train_ind,])

#roughly 15.28% have heart disease
sum(train$TenYearCHD)/2742


# Create testing data set
h_test = ht[-train_ind,]

#roughly 15.2% have heart disease
sum(h_test$TenYearCHD)/914

##################################CLASSIFICATION SECTION#############################################################
############################CLASSIFICATION FOR RISK OF HEART DISEASE#########################
######Finding best covariates for logistic regression using forward and backward stepwise#####
heartmod <- glm(TenYearCHD~1, data=ht)
model=stepAIC(heartmod, direction="both", 
              scope=(~male+age+education+currentSmoker+cigsPerDay+
                       BPMeds+prevalentStroke+prevalentHyp+diabetes+
                       totChol+sysBP+diaBP+BMI+heartRate+glucose ), trace = FALSE) 

summary(model)
##########Generating logistic regression on the training data set#########
###K-Cross Validation Approach (k=5) for logistic regression####
set.seed(123)
cv_error5 = rep(0, 5)

glm_fit = glm(TenYearCHD ~ male + age + 
                cigsPerDay + sysBP, data = train)


# Note the value of K is the number of groups which the data should be split to 
#estimate the CV error, by default K=n
cv_error5 = cv.glm(train, glm_fit, K = 5)$delta[1]
#.1182

###K-Cross Validation Approach (k=10) for logistic regression###
set.seed(123)
cv_error10 = rep(0, 10)

# Note the value of K is the number of groups which the data should be split to 
#estimate the CV error, by default K=n
cv_error10 = cv.glm(train, glm_fit, K = 10)$delta[1]
#.1182

###K-fold Cross Validation Approach (k=n) for logistic regression###
set.seed(123)
cv_errorN = rep(0, 2)

# Note the value of K is the number of groups which the data should be split to 
#estimate the CV error, by default K=n
cv_errorN = cv.glm(train, glm_fit)$delta[1]
#.1182

##now running cv-approved logistic regression on the testing dataset
glm_fit = glm(
  TenYearCHD ~ male + age + cigsPerDay + sysBP,
  data = train,
  family = binomial
)

summary(glm(
  TenYearCHD ~.,
  data = train,
  family = binomial
))


# Predicted probabilities for the testing data set
glm_probs = predict(glm_fit, h_test, type="response")

# Sample size for the testing data
dim(h_test)

# For predicted probabilities greater than 0.5, assign Y to be "Up"; otherwise assign Y to be "Down"
glm_pred = rep(0, 914) #just creates predictions all set to 0 at first
glm_pred[glm_probs > .5] = 1

# Confusion matrix
table(glm_pred, h_test$TenYearCHD)
# Proportation of correct classifications
mean(glm_pred==h_test$TenYearCHD)
#.8544 correct classification rate

# Misclassfication error rate
# glm_pred is the predicted Y for testing data and Direction_test is the true Y for testing data
mean(glm_pred!=h_test$TenYearCHD)
#.1455 misclassification error rate

###############Classification Tree for only categorical covariates########################
#ONLY TWO INTERIOR NODES, WANTS TO CLASSIFY ALL INDIVIDUALS AS NOT HAVING HEART DISEASE
#DOES IT ALREADY PRUNE?
library(tree)
## Run classification tree on the training data
tree.heart <-tree(as.factor(TenYearCHD)~., data=train)

plot(tree.heart)
text(tree.heart, pretty = 0, cex = 0.8)

## Predict the class on the test data
tree.pred <-predict(tree.heart, h_test, type="class")

## Confusion matrix
table(tree.pred, h_test$TenYearCHD)
#tree.pred   0   1
#         0 776 138
#         1   0   0

## Mis-classification error
mean(tree.pred!=h_test$TenYearCHD)
#0.1509847


#prune tree
set.seed(3)
cv.heart <-cv.tree(tree.heart, FUN = prune.misclass)
summary(cv.heart)
#use to find the best number of terminal nodes
#it is the size that minimizes the deviation!!
plot(cv.heart$size, cv.heart$dev, type="b") 

plot(cv.heart$k, cv.heart$dev, type="b")


#################linear discrimination analysis (different to logistic regression)###########
library(MASS)
lda.fit = lda(TenYearCHD ~ male + age + cigsPerDay + sysBP, data = train)
lda.pred = predict(lda.fit, h_test)
(cf.m = table(h_test$TenYearCHD, lda.pred$class))
#Creates following confusion matrix
#    0   1
#0 767   9
#1 126  12

#misclassification rate
mean(lda.pred$class!=h_test$TenYearCHD)

#################quadratic discriminant analysis##########################
qda.fit = qda(TenYearCHD ~ male + age + cigsPerDay + sysBP, data = train)
qda.pred = predict(qda.fit, h_test)
(cf.m = table(h_test$TenYearCHD, qda.pred$class))
#Creates following confusion matrix
#    0   1
#0 753  23
#1 116  22

#misclassification rate
mean(qda.pred$class!=h_test$TenYearCHD)

#################KNN#######################################################
# kNN with k=1
# Perform K-nearest neighbours on the training data set
library(class)
# Create training data for X
train.X = train[,c(1, 2, 5, 11, 14)]
# Create testing data for X
test.X = h_test[,c(1, 2, 5, 11, 14)]
# Create training data for Y
train.TenYearCHD = ht$TenYearCHD[train_ind]

# Set k=1, high flexibility, low bias.
set.seed(1234)
knn_pred = knn(train.X, test.X, train.TenYearCHD, k = 1)
table(knn_pred, h_test$TenYearCHD)
mean(knn_pred != h_test$TenYearCHD) #finding error rate
#.236323


#Generate plots of KNN for various values of K and find the optimal K
K1 = seq(1, 499, by = 2)
nk1 = length(K1)
knn.pred = as.data.frame(matrix(NA, nrow(h_test), nk1))
test.err = rep(NA, nk1)
K=1:250

for(i in K){
  knn.pred[,i] = knn(train.X, test.X, train.TenYearCHD, k = K1[i])
  test.err[i] = mean(knn.pred[,i] != h_test$TenYearCHD)
}

k.min = K[which.min(test.err)]
plot(x = 1/K, y = test.err, type = "l", log = "x", xaxt="n") 
ticks = exp(seq(log(0.002), log(1), length.out = 10))
axis(1, at = ticks, labels = round(ticks, 3))
abline(v=1/k.min, col="red")


#briefly finding error rate for kNN with optimal k
# Set k=k-min
set.seed(1234)
knn_pred = knn(train.X, test.X, train.TenYearCHD, k = k.min)
table(knn_pred, h_test$TenYearCHD)
mean(knn_pred != h_test$TenYearCHD) #finding error rate
#.1509847


#Number of positives and negatives
NP = sum(h_test$TenYearCHD == 1)
NN = sum(h_test$TenYearCHD == 0)

##########################################################################################
###################Generate the FP and TP curves for LDA, QDA and KNN#####################
###############LDA###################
lda.probs = lda.pred$posterior[,2] 
threshold = sort(unique(c(0, lda.probs))) 
nth = length(threshold)
FP1 = TP1 = rep(0, nth)
for(i in 1:nth){
  lda.pr = rep(1, nrow(h_test))
  lda.pr[lda.probs > threshold[i]] = 0
  TP1[i] = sum(h_test$TenYearCHD == 1 & lda.pr == 1) / NP # 
  FP1[i] = sum(h_test$TenYearCHD == 0 & lda.pr == 1) / NN # False Positive
}

###############QDA####################
qda.probs = qda.pred$posterior[,2] 
threshold = sort(unique(c(0, qda.probs))) 
nth = length(threshold)
FP2 = TP2 = rep(0, nth)
for(i in 1:nth){
  qda.pr = rep(1, nrow(h_test))
  qda.pr[lda.probs > threshold[i]] = 0
  TP2[i] = sum(h_test$TenYearCHD == 1 & qda.pr == 1) / NP # 
  FP2[i] = sum(h_test$TenYearCHD == 0 & qda.pr == 1) / NN # False Positive
}

###############KNN####################
set.seed(1234)
knn.pred = knn(train.X, test.X, train.TenYearCHD, k = k.min, prob = T) #where is k.min coming from!??
knn.pred = attributes(knn.pred)$prob
threshold = sort(unique(c(0, knn.pred)))
nth = length(threshold)
FP3 = TP3 = rep(0, nth)
for(i in 1:nth){
  knn.pr = rep(1, nrow(h_test))
  knn.pr[knn.pred > threshold[i]] = 0
  TP3[i] = sum(h_test$TenYearCHD == 1 & knn.pr == 1) / NP # 
  FP3[i] = sum(h_test$TenYearCHD == 0 & knn.pr == 1) / NN # False Positive
}
#Produce the plots
plot(
  x = c(0, 1),
  y = c(0, 1),
  type = "n",
  main = "ROC",
  xlab = "False Positive Rate", ylab = "True Positive Rate"
)
lines(FP1, TP1, col = "black", lwd = 3)  #LDA NOT PLOTTING!
lines(FP2, TP2, col = "green", lwd = 1.5) 
lines(FP3, TP3, col = "blue", lwd = 3) 
legend( 0,
        1,
        legend = c("LDA","QDA", "KNN"), col = c("black","green", "blue"),lwd = c(3, 1.5, 3),
        cex = 0.8
)

#Find the AUC for each of LDA, QDA and KNN
AUC = function(FP, TP){ a = head(TP, -1)
b = tail(TP, -1)
h = diff(FP)
s = sum((a + b) * h) / 2
return(s) }
#LDA AUC
(auc1 = AUC(FP1, TP1))
#0.2558598

#QDA AUC
(auc2 = AUC(FP2, TP2))
#0.2561058

#KNN AUC
(auc3 = AUC(FP3, TP3))
#0.6994013

#####CV Methods####
# 5 fold CV
library(caret)
library(e1071)

set.seed(123) 
train.control <- trainControl(method = "cv", number = 5)
# Train the model
model <- train(as.factor(TenYearCHD) ~ male + age + cigsPerDay + sysBP,
               data = train, method = "knn",
               trControl = train.control)
# Summarize the results
print(model)

############Bootstrap##############################
# Estimating the accuracy of the logistic regression

boot_fn = function(data, index) {
  coef(glm(TenYearCHD ~ male + age + 
             cigsPerDay + sysBP, data = data, subset = index))
}
# This returns the intercept and slope estimates for the logistic regression 
set.seed(1234)
boot_fn(ht, 1:2742)
boot_fn(ht, sample(2742, 2742, replace = T))

#use boot() to compute the standard errors of 1000 bootstrap estimates for the intercept and slope
boot=boot(ht, boot_fn, R = 1000) #second argument is the fitting model
#Gives following result:
#Bootstrap Statistics :
#  original        bias     std. error
#t1* -0.630621352  2.485922e-03 0.0442294297
#t2*  0.056507325 -1.616775e-04 0.0120481955
#t3*  0.007884941 -2.644463e-05 0.0007621430
#t4*  0.002322899 -2.510979e-06 0.0005439430
#t5*  0.002615351 -7.561820e-06 0.0003245095
#where
#t1 is B0-hat (intercept)
#t2 is male-hat
#t3 is age-hat
#t4 is cigsPerDay-hat
#t5 is sysBP-hat

# Compare with standard formula results for the regression coefficients in a logistic model
summary(glm(TenYearCHD ~ male + age + 
              cigsPerDay + sysBP, data = ht))$coef

#Estimate   Std. Error    t value     Pr(>|t|)
#(Intercept) -0.630621352 0.0421565905 -14.959022 3.779092e-49
#male         0.056507325 0.0121136941   4.664748 3.200713e-06
#age          0.007884941 0.0007299966  10.801339 8.583144e-27
#cigsPerDay   0.002322899 0.0005138801   4.520313 6.370665e-06
#sysBP        0.002615351 0.0002788294   9.379756 1.127372e-20
#gives the same statistics as Bootstrap, but with different standard errors.
#the errors are still very close to one another
stargazer(summary(glm(TenYearCHD ~ male + age + 
                        cigsPerDay + sysBP, data = ht))$coef)

###############################################################################################
##########Additional Classification Methods for other Categorical Variables in the Heart Dataset#############
#######FINDING PCA MODEL WITH JUST NUMERICAL COVARIATES########

#creating factor to group smoker results for all observations
ht.currentSmoker<-as.factor(ht$currentSmoker)

#creating factor to group hypertension results for all observations
ht.prevalentHyp<-as.factor(ht$prevalentHyp)

#creating pca from only numerical corvariates plus current smoker
ht.pca <- prcomp(ht[,-c(1, 6, 7, 8, 9, 16)], center = TRUE, scale. = TRUE)

summary(ht.pca)

library(devtools)
install_github("vqv/ggbiplot")#enter blank line if asked for updates to R
library(ggbiplot)

#Current Smoker#

#for the first and second PCs
ggbiplot(ht.pca, groups=ht.currentSmoker, ellipse=TRUE) +
  ggtitle("Current Smoker PCA")
#a clear split with heartRate and cigsPerDay,  and heart rate
#higher age seems to indicate no smoking


#For prevalentHyp#

#creating pca from only numerical corvariates plus prevalentHyp
ht.pca <- prcomp(ht[,-c(1, 4, 6, 7, 9, 16)], center = TRUE, scale. = TRUE)

#for the fifth and sixth PCs
ggbiplot(ht.pca, choices=c(1,2), groups=ht.prevalentHyp, ellipse=TRUE) +
  ggtitle("Prevalent Hypertension PCA")
#higher heartRate, diaBP, sysBP, BMI, totChol, glucose, and age seem to be correlated with prevalentHyp
#meanwhile higher levels of education is correlated with lowerHyp. 

##########################################################################################
#########################REGRESSIONS######################################################
######FINDING LINEAR REGRESSION MODEL for totChol#####
####WITH FORWARD STEPWISE SELECTION####
library(leaps)
K=10
set.seed(123)
folds = sample(rep(1:10, length = nrow(train))) #split into 10 folds

## Initialize an error matrix with row (10 different folds) and column (15 different predictors)
cv_errors=matrix(0, 10, 15)
## We write a for loop that performs cross-validation, in the kth fold, the elements of folds that equal k are in the test set and the remainder are in the training set
for(k in 1:10){
  fit_fwd = regsubsets(totChol~., data=train[folds!=k,], nvmax=15, method="exhaustive")
  for(i in 1:15){
    pred=predict_regsubsets(fit_fwd, train[folds==k,], id=i)
    cv_errors[k,i] =mean((train$totChol[folds==k]-pred)^2)
  }
}

## Average of the cv_error over all 10 folds
rmse_cv = sqrt(colMeans(cv_errors))

##### Plot of Root MSE vs model size and choose the optimal model size#####
plot(rmse_cv, ylab="Root MSE", xlab="Model Size", pch=15, type="b")
which.min(rmse_cv)
points(which.min(rmse_cv), rmse_cv[which.min(rmse_cv)], col="red", cex=2, pch=20)
#best is with 9 regressors

regfit_fwd = regsubsets(totChol ~ .,
                        data = train,
                        nvmax = 15,
                        method = "exhaustive")

summary(regfit_fwd)
#the fit with 9 is lm(totChol ~ male + age + education + cigsPerDay + BPMeds + diaBP + BMI + 
#                     heartRate + TenYearCHD)

#creating linear regression with this best fit
linreg<-lm(totChol ~ male + age + education + cigsPerDay + 
             BPMeds + diaBP + BMI + heartRate + TenYearCHD,
           data=train)

########K-Cross Validation Approach (k=5) for best fit stepwise linear regression#########
library(caret)

# 5 fold CV
set.seed(123) 
train.control <- trainControl(method = "cv", number = 5)

# Train the model
model <- train(totChol ~ male + age + education + cigsPerDay + BPMeds + diaBP + BMI + heartRate + TenYearCHD,
               data = train, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)
#41.59 for RMSE

#######K-Cross Validation Approach (k=10) for best subset regression########
set.seed(123) 
train.control1 <- trainControl(method = "cv", number = 10)

# Train the model
model <- train(totChol ~ male + age + education + cigsPerDay + BPMeds + diaBP + BMI + heartRate + TenYearCHD,
               data = train, method = "lm",
               trControl = train.control1)
# Summarize the results
print(model)
#41.43 for RMSE and very low R2

#######K-Cross Validation Approach (k=n) for best subset regression#######
set.seed(123) 
train.control2 <- trainControl(method = "LOOCV")

# Train the model
model <- train(totChol ~ male + age + education + cigsPerDay + BPMeds + diaBP + BMI + heartRate + TenYearCHD,
               data = train, method = "lm",
               trControl = train.control2)
# Summarize the results
print(model)
#41.52 for RMSE and very low R2

######finding MSE for testing data with best subset regression######

#using it to predict h_test
pred_test=predict(linreg, newdata = h_test)

#finding MSE
testEr_linreg = mean((h_test$totChol - pred_test) ^ 2)

#finding RMSE
sqrt(testEr_linreg)
#43.02227


###############Building Linear Regression Model with Ridge Regression###########
library(glmnet)
set.seed(1234)

#creating matrix of our testing and training data
x_train <-model.matrix(totChol~.-totChol, data=train)[,-1]
x_test <- model.matrix(totChol~.-totChol, data=h_test)[,-1]
y_train <-train$totChol
y_test<-h_test$totChol

#fitting ridge regression
fit.ridge <-glmnet(x_train, y_train, alpha=0)

## k-fold cross validation to determine the optimal tuning parameter, lambda. By default k is set to be 10
cv.ridge <-cv.glmnet(x_train, y_train, alpha=0)
## Plot of CV mse vs log (lambda)
plot(cv.ridge)
## Coefficent vector corresponding to the mse which is within one standard error of the lowest mse using the best lambda.
coef(cv.ridge)
## Coefficient vector corresponding to the lowest mse using the best lambda
coef(glmnet(x_train,y_train,alpha=0, lambda=cv.ridge$lambda.min))#alpha=0 corresponds to ridge penalty
#s0
#(Intercept)     119.40693860
#male             -5.25428747
#age               1.18605643
#education         1.86468818
#currentSmoker    -0.64620376
#cigsPerDay        0.20534072
#BPMeds            6.76752364
#prevalentStroke  -4.31339685
#prevalentHyp     -0.13579801
#diabetes          5.70380390
#sysBP             0.06228683
#diaBP             0.21735342
#BMI               0.72438663
#heartRate         0.16696118
#glucose          -0.03568468
#TenYearCHD        4.71864841

ridge_lam=cv.ridge$lambda.min
#3.313

#finding RMSE on testing data
ridge_pred = predict(fit.ridge, s = ridge_lam, newx = x_test)
sqrt(mean((y_test-ridge_pred)^2))
#RMSE:42.98

###################Building Linear Regression Model with Lasso#######################
set.seed(123)
fit.lasso <-glmnet(x_train,y_train, alpha=1)
plot(fit.lasso, xvar="lambda", label= TRUE)
plot(fit.lasso, xvar="dev", label= TRUE)
cv.lasso <-cv.glmnet(x_train, y_train, alpha=1)
plot(cv.lasso)

#finding lambda that minimizes the training MSE???
bestlambda=cv.lasso$lambda.min

abline(v=log(bestlambda), col="blue", lwd=2)

#finds coefficient of best fit on training data
lasso_coef<-predict(fit.lasso, type = "coefficients", s = bestlambda)[1:16,]
lasso_coef[lasso_coef != 0]
#is this the best lambda though?

#Coefficients for best fit on training data
#(Intercept)            male             age       education      cigsPerDay          BPMeds 
#116.473012229    -4.944001292     1.263081419     1.713476629     0.180060082     5.553316758 
#prevalentStroke        diabetes           sysBP           diaBP             BMI       heartRate 
#-0.663877127     2.176077275     0.034039275     0.243472329     0.699309483     0.159460263 
#glucose      TenYearCHD 
#-0.009025532     4.001483032 

# Use best lambda for lasso model to predict test data
lasso_pred = predict(fit.lasso, hs = bestlambda, newx = x_test)
#Calculate test RMSE
lasso_rmse=sqrt(mean((y_test - lasso_pred)^2))
#43.24
#slightly better than ridge regression


################Building Linear Regression Model with Elastic Net####################
set.seed(123)
cv_5 = trainControl(method = "cv", number = 5)

chol_elnet = train(
  totChol ~ ., data = train,
  method = "glmnet",
  trControl = cv_5)

#Optimal alpha is 0.1, between ridge and lasso

chol_elnet_int = train(
  totChol ~ . ^ 2, data = train,
  method = "glmnet",
  trControl = cv_5,
  tuneLength = 10
)

get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}
get_best_result(chol_elnet_int)

set.seed(123)
X = model.matrix(totChol ~ . ^ 2, train)[, -1]
y = train$totChol

fit_elastic_cv = cv.glmnet(X, y, alpha = 1)
elastic_rmse=sqrt(fit_elastic_cv$cvm[fit_elastic_cv$lambda == fit_elastic_cv$lambda.min])
#41.1455

#############################RANDOM FOREST#######################################
set.seed(123)
chol_bag = randomForest(totChol ~ male + age + education + 
                          cigsPerDay + BPMeds + diaBP + 
                          BMI + heartRate + TenYearCHD, data = ht, mtry = 9, 
                        importance = TRUE, ntrees = 1000)
chol_bag

chol_bag_tst_pred = predict(chol_bag, newdata = h_test)
plot(chol_bag_tst_pred,h_test$totChol,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs Actual: Bagged Model, Test Data",
     col = "dodgerblue", pch = 20)
grid()
abline(0, 1, col = "darkorange", lwd = 2)
bag_tst_rmse = sqrt(mean((chol_bag_tst_pred-h_test$totChol)^2)) 
####################################################################################
######################End of Code###################################################