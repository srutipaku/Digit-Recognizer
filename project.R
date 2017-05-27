library(readr)
library(caTools)
library(h2o)
library(caret)
library(randomForest)
library(deepnet)
library(ggplot2)
library(reshape2)
library(grid)
library(readr)
set.seed(101)

#Read Data
data <- read.csv("C:/Users/pakusruti/Downloads/train.csv")

###Checking for NA values in dataset
apply(data, MARGIN = 2, FUN = function(x) sum(is.na(x)))

##### Printing function ######
printing<- function(filename,outputFile){
  pdf(filename,width = 20,height = 20)
  par(mfrow=c(5,5))
  matrix_list = matrix(unlist(test[25]),nrow = 28,byrow = T)
  image(matrix_list,col=grey.colors(255))
  rotate <- function(x) t(apply(x, 2, rev)) # reverses (rotates the matrix)
  lapply(1:25, 
         function(x) image(
           rotate(matrix(unlist(test[x,-1]),nrow = 28,byrow = T)),
           col=grey.colors(255),
           xlab=outputFile[x,1]
         )
  )
  dev.off()
}

#Splitting the labels and the data and splitting the data set into 80% train data and 20% test data
splitdata = sample(1:nrow(data), size=0.8*nrow(data))
train <-data[splitdata,]
test <- data[-splitdata,]
labels_train <-train[,1]
labels_test<- test[,1]

printing("input.pdf",test)

#Implementation of DeeplLearning classification using H2o package

paramH2o = h2o.init(max_mem_size = '7g', nthreads = 1)
train[,1]=as.factor(train[,1])
test[,1]=as.factor(test[,1])
h2o_train = as.h2o(train)
h2o_test = as.h2o(test)

deep_model =
  h2o.deeplearning(x = 2:785,  
                   y = 1,
                   training_frame = h2o_train, 
                   activation = "RectifierWithDropout",
                   input_dropout_ratio = 0.2, 
                   hidden_dropout_ratios = c(0.5,0.5), 
                   balance_classes = TRUE, 
                   hidden = c(150,150),
                   momentum_stable = 0.99,
                   nesterov_accelerated_gradient = T, 
                   epochs = 15)

#h2o.confusionMatrix(deep_model)
h2o_test_result <- h2o.predict(deep_model, h2o_test)
dataFrame_test = as.data.frame(h2o_test_result)
dataFrame_test = data.frame(ImageId = seq(1,length(dataFrame_test$predict)), Label = dataFrame_test$predict)
write.csv(dataFrame_test, file = "h2o_prediction.csv", row.names=F)

output <- read.csv("h2o_prediction.csv",header=TRUE)
print("Accuracy of DeepLearning:" )
confusionMatrix(test[,1],output[,2])
printing("output.pdf",output)
h2o.shutdown(prompt = F)

### mplementation of DeepLearning using DeepNet package
train_net <-data[splitdata,]
test_net <- data[-splitdata,]
train.count <- nrow(train_net)
test.count <- nrow(test_net)
input.x <- as.matrix(train_net[1:train.count, -1]) / 255
input.y <- as.matrix(train_net[1:train.count, 1])
output.x <- as.matrix(test_net[1:test.count,-1]) / 255
test.y<- as.matrix(test_net[1:test.count, 1])
dnn <- dbn.dnn.train(input.x, input.y, hidden = c(100, 70, 80), numepochs = 3, cd = 3)
output.y <- nn.predict(dnn, output.x)
err.dnn <- nn.test(dnn,output.x,test.y)
print("Accuracy of deepnet:" )
print(100 - err.dnn)

# Implementation of Random Forest##.

numTrees <- 25
labels <- as.factor(train[,1])
train <- train[,-1]
labels_test<-test[,1]
test<- test[,-1]
rf <- randomForest(train, labels, xtest=test, ntree=numTrees)
predictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])
head(predictions)
print("Random Forest Accuracy: ")
confusionMatrix(labels_test,predictions[,2])

