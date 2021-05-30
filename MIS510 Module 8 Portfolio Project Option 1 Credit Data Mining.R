#Import Dataset
GermanCredit.df <- read.csv("GermanCredit.csv")

#Exploratory Data Functions
#Summary of GermanCredit.csv
summary(GermanCredit.df)

#Sum of the Missing Values
data.frame("Missing Values" = sapply(GermanCredit.df, function(x){
  sum(length(which(is.na(x))))}))

#Counts of Binary Values
BinaryValues <- GermanCredit.df[, -c(1:4, 11:14, 20, 23, 27, 28, 29)]
data.frame("Yes" = sapply(BinaryValues, function(x){
  sum(x=="1")}),
  "No" = sapply(BinaryValues, function(x){
    sum(x=="0")}))

#Means and Standard Deviations of Numerical Variables
NumericalVariables <- GermanCredit.df[, -c(1:2, 4:10, 12, 13, 15:22, 24:26, 28,
                                           30:32)]
data.frame("Standard Deviation" = sapply(NumericalVariables, sd),
           "Mean" = sapply(NumericalVariables, mean))

#Drop Observation Column for Prediction Models
GermanCredit.df <- GermanCredit.df[, -1]

#Dividing the GermanCredit Dataset into Training and Validation Sets
set.seed(1)
train.index <- sample(c(1:dim(GermanCredit.df)[1]), 
                      dim(GermanCredit.df)[1]*0.6)
train.df <- GermanCredit.df[train.index,]
valid.df <- GermanCredit.df[-train.index,]

#Pruning a Classification Tree
library(rpart)
library(rpart.plot)
class.tree <- rpart(RESPONSE ~ ., data = train.df,
                    method = "class",
                    cp = 0.0001, minsplit = 5, xval = 5)
printcp(class.tree)

#Creating Pruned Classification Tree
class.tree <- rpart(RESPONSE ~ ., data = train.df,
                    control = rpart.control(maxdepth = 3),
                    method = "class")
prp(class.tree, type = 1, extra = 1, split.font = 1, varlen = -10)

#Assessing Classification Tree Performance
#Classification Tree Training Data Performance
library(lattice)
library(ggplot2)
library(caret)
class.tree.point.pred.train <- predict(class.tree, 
                                       train.df, type ="class")
train.df$RESPONSE <- factor(train.df$RESPONSE,
                            levels = c(0,1))
confusionMatrix(class.tree.point.pred.train, train.df$RESPONSE, 
                positive = '1')

#Classification Tree Validation Data Performance
class.tree.point.pred.valid <- predict(class.tree,
                                       valid.df, type = "class")
valid.df$RESPONSE <- factor(valid.df$RESPONSE,
                            levels = c(0,1))
confusionMatrix(class.tree.point.pred.valid, valid.df$RESPONSE, 
                positive = '1')

library(pROC)
PredicProbTree <- predict(class.tree, valid.df, type = 'prob')
auc <- auc(valid.df$RESPONSE, PredicProbTree[, 2])
plot(roc(valid.df$RESPONSE, PredicProbTree[,2]), main = "Classification ROC Curve
     (AUC = 0.7549)") 
auc(valid.df$RESPONSE, PredicProbTree[, 2])

#Preprocessing GermanCredit.csv for Input into Artificial Neural Network
#Normalizing Non-Binary Variables
GermanCredit.df <- read.csv("GermanCredit.csv")
library(BBmisc)
GermanCredit.df <- GermanCredit.df[, -1]
GermanCreditNormalized.df <- normalize(GermanCredit.df, 
                             method = "range",
                              range = c(0,1),
                              margin = 1)

#Building Training and Validation Sets for Artificial Neural Network
library(neuralnet)
library(nnet)
vars=c("CHK_ACCT", "DURATION", "HISTORY", "NEW_CAR", "USED_CAR", "FURNITURE",
       "RADIO.TV", "EDUCATION", "RETRAINING", "AMOUNT", "SAV_ACCT",
       "EMPLOYMENT", "INSTALL_RATE", "MALE_DIV", "MALE_SINGLE",
       "MALE_MAR_or_WID", "CO.APPLICANT", "GUARANTOR", "PRESENT_RESIDENT",
       "REAL_ESTATE", "PROP_UNKN_NONE", "AGE", "OTHER_INSTALL", "RENT",
       "OWN_RES", "NUM_CREDITS", "JOB", "NUM_DEPENDENTS", "TELEPHONE",
       "FOREIGN")
set.seed(2)
training = sample(row.names(GermanCreditNormalized.df), dim(
  GermanCreditNormalized.df)[1]*0.6)
validation = setdiff(row.names(GermanCreditNormalized.df), training)
trainData <- cbind(GermanCreditNormalized.df[training, c(vars)],
                   class.ind(GermanCreditNormalized.df[training,]$RESPONSE)) 
names(trainData)=c(vars, paste("RESPONSE", c(0,1), sep = ""))

validData <- cbind(GermanCreditNormalized.df[validation, c(vars)],
                   class.ind(GermanCreditNormalized.df[validation,]$RESPONSE))
names(validData)=c(vars, paste("RESPONSE", c(0,1), sep = ""))

#Building the Artificial Neural Net
ANN <- neuralnet(RESPONSE0 + RESPONSE1 ~ ., data = trainData, linear.output = F,
                 hidden = 6)
plot(ANN, rep = "best")

#Assessing the Performance of the Artificial Neural Net
#Classification Accuracy and Model Performance on Training Data
training.prediction = compute(ANN, trainData[, -c(31:32)])
training.class = apply(training.prediction$net.result, 1, which.max)-1
training.class <- factor(training.class,
                            levels = c(0,1))
GermanCreditNormalized.df$RESPONSE <- factor(GermanCreditNormalized.df$RESPONSE,
                                   levels = c(0,1))
confusionMatrix(training.class, GermanCreditNormalized.df[training,]$RESPONSE,
                positive = "1")

#Classification Accuracy and Model Performance on Validation Data
validation.prediction = compute(ANN, validData[,-c(31:32)])
validation.class=apply(validation.prediction$net.result,1,which.max)-1
validation.class <- factor(validation.class,
                           levels = c(0,1))
confusionMatrix(validation.class, GermanCreditNormalized.df[validation,]$RESPONSE,
                positive = "1")

PredicProbANN <- predict(ANN, validData, type = 'prob')
aucANN <- auc(validData$RESPONSE0, PredicProbANN[, 2])
plot(roc(validData$RESPONSE0, PredicProbANN[,2]), 
     main = "Artificial Neural Net Variable Response0
     ROC Curve (AUC = 0.6643)")
auc(validData$RESPONSE0, PredicProbANN[,2])

aucANN2 <- auc(validData$RESPONSE1, PredicProbANN[, 2])
plot(roc(validData$RESPONSE1, PredicProbANN[,2]),
     main = "Artificial Neural Net Variable Response1
     ROC Curve(AUC = 0.6643)")
auc(validData$RESPONSE1, PredicProbANN[,2])

#Analysis of Classifier Models
"The Classification tree showed better overall classifier performance
on the validation dataset when compared to the Artificial Neural Network.
Credit Worthiness was determined to be the variable RESPONSE; which, had a
binary output, where 1 equaled good credit and 0 equaled bad credit. The
performance of the models for assessing this variable can be seen by 
comparing the overall Prediction Accuracy produced by the Confusion Matrices. 
The Classification tree had an overall Accuracy of 74.75% on the validation 
dataset. Comparatively, the Artificial Neural Net had an Accuracy of 67% on 
the validation dataset.Further confirmation that the Classification Tree 
outperformed the Artifical Neural Network can also be seen from the produced 
ROC Curves. Where the Classification Tree produced a value of 0.7549 Area 
Under the Curve versus 0.6643 produced by the Neural Net."

