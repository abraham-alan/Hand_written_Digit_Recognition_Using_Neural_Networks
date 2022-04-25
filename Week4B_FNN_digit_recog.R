#Predicting Hand_Written Digits Using 7 Segment-Combination (Based on LED Digit Display)----

setwd("~/ALY-6020")
# read in the 2 datasets for training and testing
mnist_test <- read.csv("mnist_test.csv", header = FALSE)
mnist_train <- read.csv("mnist_train.csv", header = FALSE)

#add 7 columns, 1 per digit defined by its 7-segment combination
mnist_train$seg1 <- ifelse(mnist_train$V1 %in% c("0","2","3","5","6","7","8","9"),1,0)
mnist_train$seg2 <- ifelse(mnist_train$V1 %in% c("0","2","3","4","1","7","8","9"),1,0)
mnist_train$seg3 <- ifelse(mnist_train$V1 %in% c("0","1","3","4","5", "6","7","8","9"),1,0)
mnist_train$seg4 <- ifelse(mnist_train$V1 %in% c("0","2","3","5","6","8"),1,0)
mnist_train$seg5 <- ifelse(mnist_train$V1 %in% c("0","2","6","8"),1,0)
mnist_train$seg6 <- ifelse(mnist_train$V1 %in% c("0","4","5","6","8","9"),1,0)
mnist_train$seg7 <- ifelse(mnist_train$V1 %in% c("2","3","5","6","4","8","9"),1,0)

#view if added properly
mnist_train[1:20,c(1,786:792)]

install.packages("Rtools")
install.packages("RSNNS")
library(RSNNS)
#set x and y variables for training data
digits.x <- mnist_train[,-c(1,786:792)]
digits.y <- mnist_train[,786:792]

#set x and y variables for testing data
digits.test.x <- mnist_test[,-1]
digits.test.y <- mnist_test[,1]

# normalize both data sets of x
digits.x <- normalizeData(digits.x)
digits.test.x <- normalizeData(digits.test.x)

set.seed(42)
tic <- proc.time()
digits.model <- mlp(as.matrix(digits.x),
                    digits.y, 
                    size = 40,             
                    learnFunc = "Rprop",             
                    shufflePatterns = FALSE,
                    maxit = 80)
print(proc.time() - tic)
# took an hour!
plotIterativeError(digits.model)

digits.yhat <- predict(digits.model ,newdata=digits.test.x)

digits.yhat
digits.yhat2 <- round(digits.yhat)

#convert back to numbers and then test for accuracy
# first create interpretation model by building a matrix
segment7 <- matrix(c(
  1,1,1,1,1,1,0,
  0,1,1,0,0,0,0,
  1,1,0,1,1,0,1,
  1,1,1,1,0,0,1,
  0,1,1,0,0,1,1,
  1,0,1,1,0,1,1,
  1,0,1,1,1,1,1,
  1,1,1,0,0,0,0,
  1,1,1,1,1,1,1,
  1,1,1,0,0,1,1),10, 7, byrow=TRUE)

digits.yhat2 <- as.data.frame(digits.yhat2)
digits.yhat2[1,1:7]
# convert back to a digit based on the 7-segment combinations
results <- rep(NA,10000)
for (i in 1:nrow(digits.yhat2)) {
  
results[i] <- ifelse(all(digits.yhat2[i,] == segment7[1,]), 0,
            ifelse(all(digits.yhat2[i,] == segment7[2,]), 1,
            ifelse(all(digits.yhat2[i,] == segment7[3,]), 2,
            ifelse(all(digits.yhat2[i,] == segment7[4,]), 3,
            ifelse(all(digits.yhat2[i,] == segment7[5,]), 4,
            ifelse(all(digits.yhat2[i,] == segment7[6,]), 5,
            ifelse(all(digits.yhat2[i,] == segment7[7,]), 6,
            ifelse(all(digits.yhat2[i,] == segment7[8,]), 7,
            ifelse(all(digits.yhat2[i,] == segment7[9,]), 8,
            ifelse(all(digits.yhat2[i,] == segment7[10,]), 9,10))))))))))
}


# if didn't match then assign a "10"
table(results)

barplot(table(results),main="Distribution of y values (ANN model)", col = 1:6)
sum(is.na(results))
RSNNS::confusionMatrix(mnist_test$V1, results)

library(caret)
attributes(mnist_test$V1)
attributes(results)
mnist_test$V1 <- as.factor(mnist_test$V1)
results <-as.factor(results)
confm.digit <- caret::confusionMatrix(mnist_test$V1, results)
confm.digit
# ranging from 93 - 99% accuracy!!!
sum(confm.digit$table[,3])
(906+1063+880+875+886+707+883+907+799+799+871)/10000
632+9576
