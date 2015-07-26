# Machine Learning. Johns Hopkins course project.
Denis Pluzhnikov  
Sunday, July 26, 2015  

##Synopsis
The goal of this project is to predict exactly in which way barbell was lifted out of 5 possible ways by using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. PCA was used for preprocessing and Random forrest with cross validation for modelling. 
Data was separated into training set for modelling and testing set for validation. Separation ratio is 60% to 40% respectively.

Data used for the project are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

Detailed description on data is here:[http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har)

##Data processing
Data was downloaded and read into R dataframe.

```r
#load all libraries required
library(caret)
library(dplyr)
library(kernlab)
library(pander)

# setwd("C:/Users/Tamagoch/Desktop/Git/MachineLearning/Ass1/")
#set url and download data
url<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
if(!file.exists("./data")){dir.create("./data")}
download.file(url,"./data/train.csv")

#read data into R data frame
train<-read.csv("./data/train.csv",)
```

Some columns(features) were dropped out if one of the following conditions were met:    
*  more then 19000(95%) of empty(NA) values  
*  insufficient variability(caret nearZeroVar function)  
*  feature is inappropriate for prediction matter(observation ID, timestamps, window id)  

User name(name of participant) was transformed into dummy variable for further PCA preProcessing
58 eligable for prediction variables left as the result.

Then cleaned data was separated into training(60%) and validation(40%) samples. 

```r
#drop empty cols
na_clear<-data.frame(name=names(train),cnt=colSums(is.na(train)))
train_clear<-train[,as.character(na_clear[na_clear$cnt<19000,1])]

#drop cols with nearly zero variability
zerovar<-nearZeroVar(train_clear,saveMetrics = T)
train_clear <- train_clear[,rownames(zerovar[zerovar$nzv==FALSE,])]

#create dummy feature for user_name
dummies<- dummyVars(classe ~ user_name,data=train_clear)
train_clear <- cbind(train_clear, data.frame( predict(dummies,newdata=train_clear)))

#drop observation ID, timestamps, window id and old user_name col
train_clear<-train_clear[,c(-1,-2,-3,-4,-5,-6)]

#set 60% training and 40% validation sample
set.seed(seed = 1894)
inTrain = createDataPartition(train_clear$classe, p = 0.6,list=F)
training = train_clear[ inTrain,]
testing = train_clear[-inTrain,]
```


Taking into consideration large number of potential predictors and higl level of in-between correlation Principal Component Analysis(PCA) 
with prior normalization was chosen as major preprocessing approach. Centering and scaling used as methods for normalization.
85% threshold level for PCA is set to avoid overfitting.  
As result of PCA preprocess the **number of predictors decreased to the total of 16.**

```r
#preprocess with normalization
preObj1<-preProcess(training[,-53],method = c("center","scale"))
training_pre1<-predict(preObj1,training[,-53])
testing_pre1<-predict(preObj1,testing[,-53])

#preprocess with PCA
preObj2<-preProcess(training_pre1,method="pca",thresh = 0.85)
training_pre2<-predict(preObj2,training_pre1)
testing_pre2<-predict(preObj2,testing_pre1)
```

##Model training

To build a model Random Forrest was used, with **cross validation(5 folds) resampling method as training control.** 
Training was processed on training data sample exclusively

```r
#set training control to cross validation with 5 folds
ctrl<-trainControl(method = "cv", number = 5)
training_final<-cbind(training_pre2,classe=training$classe)

#fit the model
set.seed(seed = 1894)
model<-train(classe~.,data=training_final,method = "rf", trControl = ctrl)
```

##Model validation

To validate the model "Classe" outcome for testing data set was predicted and compared to actual outcome.
correcly predicted values may be seen on the diagonal fo matrix bellow.

```r
pred<-predict(model,testing_pre2)

pander(confusionMatrix(pred,testing$classe)$table)
```


--------------------------------
&nbsp;   A    B    C    D    E  
------- ---- ---- ---- ---- ----
 **A**  2178  41   17   9    3  

 **B**   15  1445  16   4    12 

 **C**   25   29  1319  61   11 

 **D**   12   3    13  1211  13 

 **E**   2    0    3    1   1403
--------------------------------

```r
pander(data.frame(Value = confusionMatrix(pred,testing$classe)$overall))
```


------------------------------
       &nbsp;          Value  
-------------------- ---------
    **Accuracy**       0.963  

     **Kappa**        0.9532  

 **AccuracyLower**    0.9586  

 **AccuracyUpper**    0.9671  

  **AccuracyNull**    0.2845  

 **AccuracyPValue**      0    

 **McnemarPValue**   2.886e-12
------------------------------

**Total accuracy for testing data sample is  0.96.**
**And out-of-sample error is:**

```r
1-round(confusionMatrix(pred,testing$classe)$overall[[1]],2)
```

```
## [1] 0.04
```

