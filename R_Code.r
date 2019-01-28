# 1. Business Understanding: 

# THE OBJECTIVE IS TO BUILD A MODEL TO IDENTIFY THE NUMBER THROUGH SOME 785 PIXEL ATTRIBUTE AVAILABLE

#####################################################################################

# 2. Data Understanding: 
# Taking 15 percent of entire train data provided 
# Number of Instances: 9000
# Number of Attributes: 785

#3. Data Preparation: 


#Loading Neccessary libraries

library(kernlab)
library(readr)
library(caret)


#Loading Data

Data <- read.csv("mnist_train.csv",header = FALSE,stringsAsFactors = FALSE)

test <- read.csv("mnist_test.csv",header = FALSE,stringsAsFactors = FALSE)

#Coverting the first column into factorial

Data$V1<-as.factor(Data$V1)

test$V1<-as.factor(test$V1)

#Understanding Dimensions

dim(Data)

#Structure of the dataset

str(Data)

#printing first few rows

head(Data)

#Exploring the data

summary(Data)

#checking missing value

sapply(Data, function(x) sum(is.na(x)))





# Split the data into train and test set

set.seed(1)

#Taking 15 percent data as suggested
train.indices = sample(1:nrow(Data), 0.15*nrow(Data))
train = Data[train.indices, ]



#Constructing Model

#Using Linear Kernel
Model_linear <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, test)




#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,test$V1)
#Accuracy came as 0.9095 with C=1

#Using RBF Kernel
Model_RBF <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, test)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,test$V1)

# Accuracy : 0.9564

#Printing RBF model to find out the sigma and cost
print(Model_RBF)

#Support Vector Machine object of class "ksvm" 

#SV type: C-svc  (classification) 
#parameter : cost C = 1 

#Gaussian Radial Basis kernel function. 
#Hyperparameter : sigma =  1.64691361853384e-07 


############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 5 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model,We will pass the sigma value between uper and 
# and lower range of sigma value obtained from RBF model

set.seed(7)
grid <- expand.grid(.sigma=c(1.00001361853384e-07,1.64691361853384e-07, 2.00001361853384e-07), .C=c(0.5,1,1.5) )


#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(V1~., data=train, method="svmRadial", metric=metric, 
                                                   tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)



#Summary of sample sizes: 7201, 7199, 7201, 7200, 7199 
#Resampling results across tuning parameters:
  
#  sigma         C    Accuracy   Kappa    
#1.000014e-07  0.5  0.9355567  0.9283559
#1.000014e-07  1.0  0.9428886  0.9365077
#1.000014e-07  1.5  0.9473331  0.9414502
#1.646914e-07  0.5  0.9442226  0.9379909
#1.646914e-07  1.0  0.9519999  0.9466383
#1.646914e-07  1.5  0.9570001  0.9521973
#2.000014e-07  0.5  0.9471111  0.9412029
#2.000014e-07  1.0  0.9556666  0.9507152
#2.000014e-07  1.5  0.9597775  0.9552850

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were sigma = 2.000014e-07 and C = 1.5.

#Checking with sigma 2.000014e-07 and c=1.5
Eval_RBF<- predict(fit.svm, test)
confusionMatrix(Eval_RBF,test$V1)
#      Accuracy : 0.9623  
final_model<-fit.svm
