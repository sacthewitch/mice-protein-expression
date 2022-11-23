install.packages('Amelia')
# required libraries
require(readxl)
require(data.table)
require(caret)
require(xgboost)
require(corrplot)
require(dplyr)
require(randomForest)
require(ipred)
require(gbm)
require(mlbench)
require(e1071)
require(doParallel)
require(rpart.plot)
require(Amelia)
require(ggplot2)

# loading the data
data_cortex <- read_excel('Data_Cortex_Nuclear.xls')

# get to know the data
summary(data_cortex)
colnames(data_cortex)
dim(data_cortex)

# missing Values
dim(na.omit(data_cortex))

# nearly 50% of observations contains missing values. We cannot remove NA values.



# Step 1: Treating Missing Values ----

# visualize missingness
missmap(data_cortex)


# Removing 'MouseID', 'Treatment', 'Behavior' attributes
new_micedata <- data_cortex[,-c(1,80,81)]

# Impute missing values with mean value with respect to its corresponding class
new_micedata <- new_micedata %>%
  group_by(class) %>%
  mutate_each(funs(replace(., which(is.na(.)), mean(., na.rm=TRUE)))) %>%
  as.data.frame()

# New dataset
summary(new_micedata)

# sampling 30/70
DataIndex <- createDataPartition(new_micedata$class, p=0.7, list = FALSE)

# Training Testing Subset
training_data <- new_micedata[DataIndex,]
test_data <- new_micedata[-DataIndex,]

#training controls
controls = trainControl(method = "repeatedcv", number = 10, repeats = 3)

#Test model with Parallel for speed
# Calculate the number of cores
no_cores <- detectCores() - 1
no_cores
cl <- makeCluster(no_cores)
registerDoParallel(cl)



# Use the 77 proteins as predictors for decision trees and 
# support vector machines models to make binary and multiple class classification----

## Binary class classification ----

### Decision Tree ----
set.seed(23)
registerDoSEQ()
bi_dtree_model <- train(x = training_data[,-c(79)], 
                     y = training_data$Genotype,
                     method = "rpart",
                     parms = list(split = "information"),
                     trControl=controls,
                     tuneLength = 10)


# View decision tree details
bi_dtree_model

# decision tree plot
prp(bi_dtree_model$finalModel, box.palette = "Blues", tweak = 1.2)

# Test decision tree model on test data
bi_prediction_dtree <- predict(bi_dtree_model, test_data[,-c(79)])

# Change class data to factor
test_data$Genotype = as.factor(test_data$Genotype)
str(test_data$Genotype)

# accuracy
confusionMatrix(bi_prediction_dtree, test_data$Genotype)

# See which proteins explained the most variance
plot(varImp(bi_dtree_model), top = 35)

### Support vector machine ----
set.seed(23)

#Change class data to factor
training_data$Genotype = as.factor(training_data$Genotype)
str(training_data$Genotype)


# SVM fit
bi_svm_model <-  svm(training_data$Genotype ~ . , data =  training_data[,-c(79)],  
                  kernel = "linear", type = "C-classification", cost = 1 , 
                  scale = FALSE)

# View SVM details
bi_svm_model

# Test SVM model on test data
bi_prediction_svm <- predict(bi_svm_model, test_data[,-c(79)])

# accuracy
confusionMatrix(bi_prediction_svm, test_data$Genotype)



## Multiple class classification ----

### Decision Tree ----
set.seed(23)
dtree_model <- train(x = training_data[,-c(78)], 
                   y = training_data$class,
                   method = "rpart",
                   parms = list(split = "information"),
                   trControl=controls,
                   tuneLength = 10)


# View decision tree details
dtree_model

# decision tree plot
prp(dtree_model$finalModel, box.palette = "Blues", tweak = 1.2)

# Test decision tree model on test data
prediction_dtree <- predict(dtree_model, test_data[,-c(78)])

# Change class data to factor
test_data$class = as.factor(test_data$class)
str(test_data$class)

# accuracy
confusionMatrix(prediction_dtree, test_data$class)

# See which proteins explained the most variance
plot(varImp(dtree_model), top = 35)


### Support Vector Machine ----

set.seed(23)

#Change class data to factor
training_data$class = as.factor(training_data$class)
str(training_data$class)


# SVM fit
svm_model <-  svm(training_data$class ~ . , data =  training_data[,-c(78)],  
                  kernel = "linear", type = "C-classification", cost = 1 , 
                  scale = FALSE)

# View SVM details
svm_model

# Test SVM model on test data
prediction_svm <- predict(svm_model, test_data[,-c(78)])

# accuracy
confusionMatrix(prediction_svm, test_data$class)



# Perform principal component analysis on the 77 numerical features. 
# Use an appropriate number of principal components as predictors and perform the same classification task ----

# Run PCA
pca <- prcomp(new_micedata[,-c(78,79)], scale = TRUE)
names(pca)


# Summary
pca$rotation

# See the principal components
dim(pca$x)
pca$x

biplot(pca, main = "Biplot", scale = 0)

pca$sdev

# pca variance
var_pca <- pca$sdev ^ 2
var_pca

# plot pca variance
plot_varPca <- var_pca * 100 / sum(var_pca)
plot_varPca

# Plot variance explained for each principal component
plot(plot_varPca, xlab = "Principal Component",
     ylab = "Variance Explained %",
     ylim = c(0, 100), type = "b",
     main = "Scree Plot")

# Plot the cumulative proportion of variance explained
plot(cumsum(plot_varPca),
     xlab = "Principal Component",
     ylab = "Cumulative Variance Explained %",
     ylim = c(0, 100), type = "b")

# top principal components that cover 90 % variance of dimension
which(cumsum(plot_varPca) >= 90)[1] 

# new dataset
pca_micedata <- data.frame(pca$x[, 1:20], Genotype = new_micedata$Genotype, 
                            class = new_micedata$class)

# sampling 30/70
DataIndex1 <- createDataPartition(pca_micedata$class, p=0.7, list = FALSE)

# Training Testing Subset
training_data1 <- pca_micedata[DataIndex1,]
test_data1 <- pca_micedata[-DataIndex1,]


## Binary class classification ----

### Decision Tree ----
set.seed(23)
registerDoSEQ()
bi_dtree_model1 <- train(x = training_data1[,-c(22)], 
                        y = training_data1$Genotype,
                        method = "rpart",
                        parms = list(split = "information"),
                        trControl=controls,
                        tuneLength = 10)


# View decision tree details
bi_dtree_model1

# decision tree plot
prp(bi_dtree_model1$finalModel, box.palette = "Blues", tweak = 1.2)

# Test decision tree model on test data
bi_prediction_dtree1 <- predict(bi_dtree_model1, test_data1[,-c(22)])

# Change class data to factor
test_data1$Genotype = as.factor(test_data1$Genotype)
str(test_data1$Genotype)

# accuracy
confusionMatrix(bi_prediction_dtree1, test_data1$Genotype)

# See which proteins explained the most variance
plot(varImp(bi_dtree_model1), top = 20)

### Support vector machine ----
set.seed(23)

#Change class data to factor
training_data1$Genotype = as.factor(training_data1$Genotype)
str(training_data1$Genotype)


# SVM fit
bi_svm_model1 <-  svm(training_data1$Genotype ~ . , data =  training_data1[,-c(22)],  
                     kernel = "linear", type = "C-classification", cost = 1 , 
                     scale = FALSE)

# View SVM details
bi_svm_model1

# Test SVM model on test data
bi_prediction_svm1 <- predict(bi_svm_model1, test_data1[,-c(22)])

# accuracy
confusionMatrix(bi_prediction_svm1, test_data1$Genotype)



## Multiple class classification ----

### Decision Tree ----
set.seed(23)
dtree_model1 <- train(x = training_data1[,-c(21)], 
                     y = training_data1$class,
                     method = "rpart",
                     parms = list(split = "information"),
                     trControl=controls,
                     tuneLength = 10)


# View decision tree details
dtree_model1

# decision tree plot
prp(dtree_model1$finalModel, box.palette = "Blues", tweak = 1.2)

# Test decision tree model on test data
prediction_dtree1 <- predict(dtree_model1, test_data1[,-c(21)])

# Change class data to factor
test_data1$class = as.factor(test_data1$class)
str(test_data1$class)

# accuracy
confusionMatrix(prediction_dtree1, test_data1$class)

# See which proteins explained the most variance
plot(varImp(dtree_model), top = 20)


### Support Vector Machine ----

set.seed(23)

#Change class data to factor
training_data1$class = as.factor(training_data1$class)
str(training_data1$class)


# SVM fit
svm_model1 <-  svm(training_data1$class ~ . , data =  training_data1[,-c(21)],  
                  kernel = "linear", type = "C-classification", cost = 1 , 
                  scale = FALSE)

# View SVM details
svm_model1

# Test SVM model on test data
prediction_svm1 <- predict(svm_model1, test_data1[,-c(21)])

# accuracy
confusionMatrix(prediction_svm1, test_data1$class)



# Using bagging, random forest, and boosting perform the same classification task. 
# Compare the results of the three methods ----

## Binary class classification ----

### Bagging ----
bi_bag <- bagging(training_data$Genotype~., data = training_data[,-c(79)], 
                  coob = T, nbagg = 100)

# View Bagging details
bi_bag

# Test Bagging model on test data
prediction_bag <- predict(bi_bag, test_data[,-c(79)])

# accuracy
confusionMatrix(prediction_bag, test_data$Genotype)


#
### Random Forest ----
bi_rforest <- randomForest(training_data$Genotype~., data = training_data[,-c(79)], 
                           importance = TRUE, proximity = TRUE)

# View Random Forest details
bi_rforest

# Test Random Forest model on test data
prediction_rforest <- predict(bi_rforest, test_data[,-c(79)])

# accuracy
confusionMatrix(prediction_rforest, test_data$Genotype)


#
### Boosting ----
bi_boost <- gbm(training_data$Genotype ~., data = training_data[,-c(79)],
                   distribution = "multinomial", cv.folds = 10, shrinkage = .01,
                   n.minobsinnode = 10, n.trees = 500)

# View Boosting details
bi_boost

# Test Boosting model on test data
prediction_boost <- predict.gbm(object = bi_boost, newdata = test_data[,-c(79)],
                                n.trees = 500, type = "response")

boost_type = colnames(prediction_boost)[apply(prediction_boost, 1, which.max)]

# accuracy
confusionMatrix(factor(boost_type), test_data$Genotype)


#
## Multiple class classification ----

### Bagging ----
bi_bag1 <- bagging(training_data$class~., data = training_data[,-c(78)], 
                  coob = T, nbagg = 100)

# View Bagging details
bi_bag1

# Test Bagging model on test data
prediction_bag1 <- predict(bi_bag1, test_data[,-c(78)])

# accuracy
confusionMatrix(prediction_bag1, test_data$class)

#
### Random Forest ----
bi_rforest1 <- randomForest(training_data$class~., data = training_data[,-c(78)], 
                           importance = TRUE, proximity = TRUE)

# View Random Forest details
bi_rforest1

# Test Random Forest model on test data
prediction_rforest1 <- predict(bi_rforest1, test_data[,-c(78)])

# accuracy
confusionMatrix(prediction_rforest1, test_data$class)


#
### Boosting ----
bi_boost1 <- gbm(training_data$class ~., data = training_data[,-c(78)],
                distribution = "multinomial", cv.folds = 10, shrinkage = .01,
                n.minobsinnode = 10, n.trees = 500)

# View Boosting details
bi_boost1

# Test Boosting model on test data
prediction_boost1 <- predict.gbm(object = bi_boost1, newdata = test_data[,-c(78)],
                                n.trees = 500, type = "response")

boost_type1 = colnames(prediction_boost1)[apply(prediction_boost1, 1, which.max)]

# accuracy
confusionMatrix(factor(boost_type1), test_data$class)



# Use the dataset to perform clustering. try both k-means clustering and hierarchical clustering. 
# In every case, find a number of clusters that make sense and try to explain what each cluster describes ----


library(factoextra)
library(cluster)

cluster_micedata <- cbind(new_micedata, data_cortex[,c(80,81)])

#calculate gap statistic based on number of clusters
gap_stat <- clusGap(cluster_micedata[,-c(78,79,80,81)],
                    FUN = kmeans,
                    nstart = 25,
                    K.max = 12,
                    B = 50)

#plot number of clusters vs. gap statistic
fviz_gap_stat(gap_stat)

## K-means Clustering

set.seed(123)

# function to compute total within-cluster sum of square 
fviz_nbclust(cluster_micedata[,-c(78,79,80,81)], kmeans, method = "wss")


k2 <- kmeans(cluster_micedata[,-c(78,79,80,81)], centers = 2, nstart = 25)
k4 <- kmeans(cluster_micedata[,-c(78,79,80,81)], centers = 4, nstart = 25)
k8 <- kmeans(cluster_micedata[,-c(78,79,80,81)], centers = 8, nstart = 25)
k10 <- kmeans(cluster_micedata[,-c(78,79,80,81)], centers = 10, nstart = 25)

# plots to compare
p1 <- fviz_cluster(k2, geom = "point", data = cluster_micedata[,-c(78,79,80,81)]) + ggtitle("k = 2")
p2 <- fviz_cluster(k4, geom = "point",  data = cluster_micedata[,-c(78,79,80,81)]) + ggtitle("k = 4")
p3 <- fviz_cluster(k8, geom = "point",  data = cluster_micedata[,-c(78,79,80,81)]) + ggtitle("k = 8")
p4 <- fviz_cluster(k10, geom = "point",  data = cluster_micedata[,-c(78,79,80,81)]) + ggtitle("k = 10")

library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)

#find means of each cluster
aggregate(cluster_micedata[,-c(78,79,80,81)], by=list(cluster=k2$cluster), mean)
aggregate(cluster_micedata[,-c(78,79,80,81)], by=list(cluster=k4$cluster), mean)
aggregate(cluster_micedata[,-c(78,79,80,81)], by=list(cluster=k8$cluster), mean)
aggregate(cluster_micedata[,-c(78,79,80,81)], by=list(cluster=k10$cluster), mean)


#add cluster assignment to original data
final_micedata <- cbind(cluster_micedata, K2 = k2$cluster)
final_micedata <- cbind(final_micedata, K4 = k4$cluster)
final_micedata <- cbind(final_micedata, K8 = k8$cluster)
final_micedata <- cbind(final_micedata, K10 = k10$cluster)



## Hierarchical Clustering ----

 #define linkage methods
m <- c( "average", "single", "complete", "ward")
names(m) <- c( "average", "single", "complete", "ward")

# function to compute agglomerative coefficient
ac <- function(x) {
  agnes(cluster_micedata[,-c(78,79,80,81)], method = x)$ac
}

#calculate agglomerative coefficient for each clustering linkage method
sapply(m, ac)

#perform hierarchical clustering using Ward's minimum variance
clust <- agnes(cluster_micedata[,-c(78,79,80,81)], method = "ward")

#produce dendrogram
pltree(clust, cex = 0.6, hang = -1, main = "Dendrogram") 

#calculate gap statistic for each number of clusters (up to 10 clusters)
gap_stat <- clusGap(cluster_micedata[,-c(78,79,80,81)], FUN = hcut, nstart = 25, K.max = 10, B = 50)

#produce plot of clusters vs. gap statistic
fviz_gap_stat(gap_stat)

#compute distance matrix
d <- dist(cluster_micedata[,-c(78,79,80,81)], method = "euclidean")

#perform hierarchical clustering using Ward's method
final_clust <- hclust(d, method = "ward.D2" )

#cut the dendrogram into 10 clusters
groups <- cutree(final_clust, k=10)

#find number of observations in each cluster
table(groups)

#append cluster labels to original data
final_micedata <- cbind(final_micedata, HierarchicalCluster = groups)

#display first six rows of final data
head(final_micedata)



