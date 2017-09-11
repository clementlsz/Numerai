library(ggplot2)
library(caret)
library(doSNOW)
library(infotheo)
library(neuralnet)
library(e1071)
library(NeuralNetTools)
library(devtools)
library(nnet)
library(randomForest)
library(keras)
setwd("C:/Users/Lsz-C/Desktop/DS/numerai_datasets/20170906")

train = read.csv("numerai_training_data.csv", header = TRUE)
train$target = ifelse(train$target == 0, "X0", "X1")
train$target = as.factor(train$target)
train$class = 0 # 0 refers to training class

tournament = read.csv("numerai_tournament_data.csv", header = TRUE)
tournament$class = 1 # 1 refers to tournament class

classes = unlist(list(train$class,tournament$class))
classes = as.factor(ifelse(classes == 0, "X0", "X1"))

features = rbind(train[, grep("feature", names(train))], tournament[, grep("feature", names(train))])

# try to use random forest to do the classification for differernt data type (class)
features$class = classes
f_names_orig = names(features)[!names(features) %in% "class"]
set.seed(100)
rf = randomForest(x=features[,f_names_orig],y=features$class,importance = TRUE, ntree = 500)

train_predict = rf$predicted[1:nrow(train)]
#class_train_tour = train[train_predict != "X0",] # data in train set and misclassified into tournament set
#train_tour_rowcount = as.data.frame(table(class_train_tour$era))
train_tour_prob = predict(rf, train[,f_names_orig],type = 'prob')[,2] # probablity for being in the tournament set
train$tour_prob = train_tour_prob
train_in_train = train[train_predict == "X0",] # data in train set and classified into training set correctly

quantile(train_in_train$tour_prob, c(0.1,0.05))

ggplot(train_in_train,
       aes(x = tour_prob, fill = target)) +
    geom_histogram(binwidth = 0.01)

ggplot(train,
       aes(x = tour_prob, fill = target)) +
    geom_histogram(binwidth = 0.01)

sum(train_in_train$tour_prob<=0.062)

# removing the bottom percentage of training data that are least similar to the tournament data
train_remove = train_in_train[train_in_train$tour_prob<=0.062,] # adjust this probablity value to include (exclude) data points
# run some simple checks on the removed data
ggplot(train_remove,
       aes(x = tour_prob, fill = target)) +
    geom_histogram(binwidth = 0.01) # graph the distribution of the removed data
table(train_remove$era)
table(train$era)
table(train_remove$era) / table(train$era)
# create a new training data set
train_adj = train[!train$id %in% train_remove$id,]
remove(train_in_train)

# move on
tournament_valid = tournament[which(!is.na(tournament$target)),]
tournament_valid$target = ifelse(tournament_valid$target == 0, "X0", "X1")
tournament_valid$target = as.factor(tournament_valid$target)

tournament_submit = tournament[which(is.na(tournament$target)),]
remove("tournament")

features = rbind(train_adj[, grep("feature", names(train_adj))], tournament_valid[, grep("feature", names(train_adj))], tournament_submit[, grep("feature", names(train_adj))]) # scaled

features_new = matrix(0, nrow(features), 231)
column_name = vector(mode = "character", length = 231)
count = 0
# calculating polynomial terms
for (i in 1:ncol(features)) {
    for (j in i:ncol(features)) {
        count = count + 1
        features_new[, count] = features[, i] * features[, j]
        #print(paste("feature",i,"with feature",j, "count is", count))
        column_name[count] = paste(names(features)[i], as.character(j), sep =
                                       "_")
    }
}
colnames(features_new) = column_name
features_new = as.data.frame(features_new)

# scale the new feature data
maxs = apply(features_new, 2, max)
mins = apply(features_new, 2, min)
features_new = as.data.frame(scale(features_new, center = mins, scale = maxs - mins))

# join new features with the original features
features_all = cbind(features, features_new)
remove(features) # removing unused variable
remove(features_new) # removing unused variable

# run PCA on features_all
princ = prcomp(features_all) # train PCA on the all the data, including the tournament data
PCA = princ$x[,1:27] # we choose the first 27 components

training_all = as.data.frame(PCA[1:(nrow(train_adj)+nrow(tournament_valid)),])
target_all = unlist(list(train_adj$target,tournament_valid$target))
training_all$target = target_all

f = as.formula(paste("target ~", paste(names(training_all)[!names(training_all) %in% "target"] , collapse = " + ")))
numFolds =
    trainControl(
        method = 'cv',
        number = 10,
        classProbs = TRUE,
        allowParallel = TRUE
    )

set.seed(1)
cl = makeCluster(3, type = "SOCK")
registerDoSNOW(cl)
# running on the selected PCA columns
nn1 = caret::train(
    f,
    data = training_all,
    method = 'nnet',
    maxit = 1000,
    trControl = numFolds,
    tuneGrid = expand.grid(size = c(18,14,10,7), decay = c(0.1))
)

rf.1 = caret::train(x = training_all[,1:27], y = training_all[,28],method = "rf", tuneLength = 3,
                   ntree = 500, trControl = numFolds)
#rf.1$results




# this function does not have weight for the constant term... why?
# nn = caret::train(
#     f,
#     data = training_all,
#     method = 'mlpWeightDecayML',
#     maxit = 1000,
#     trControl = numFolds,
#     tuneGrid = expand.grid(layer1 = c(14,7), layer2 = 0, layer3 = 0, decay = c(0.1))
# ) 


stopCluster(cl)

PCA_valid = tail(training_all, nrow(tournament_valid))
valid_checking = predict(nn,PCA_valid)
accu = sum(valid_checking == tournament_valid$target) / length(valid_checking) # we have 52.6% accuracy

valid_predict = predict(nn,PCA_valid, type='prob')$X1
valid_predict = cbind(tournament_valid$id, valid_predict)


# on the submit data set
PCA_submit = tail(PCA,nrow(tournament_submit))
submit_predict = predict(nn,PCA_submit,type='prob')$X1
submit_predict = cbind(tournament_submit$id, submit_predict)

final_submit = rbind(valid_predict, submit_predict)
colnames(final_submit) = c("id","probability")

write.csv(final_submit,"submit20170906_part1.csv")

plotnet(nn1)
coef(nn1$finalModel)

nn$finalModel
