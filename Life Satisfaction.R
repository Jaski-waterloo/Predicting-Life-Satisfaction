library(car)
library(dplyr)
library(rpart)
library(caret)
library(gbm)
library(randomForest)
library(mlbench)
library(e1071)
library(zoo)
library(MASS)
library(FNN)
library(pROC)
library(neuralnet)
library(ranger)

##Setting data

df = read.csv('train.csv')
final = read.csv('test.csv')
df = subset(df, select = -c(id,v4,v5,v17,v25,v20,v150,v154,v155,v161,v167,v57,v59,v61,v63,v22,v5,v23,v26,v27,v28,v29,v30,v31,v32,v33,v34, v39, v45,v46,v50,v69,v78,v86,v87,v88,v89,v90,v91,v92,v93,v94,v95,v86,v88,v91,v92,v94,v95,v96,v97,v123,v124,v126, v127, v128, v129, v130, v131, v133, v151, v153,v164,v168,v174,v182,v191,v192,v193,v194,v195,v197,v198,v199,v200,v201,v202,v203,v204,v205,v206,v207,v209,v210,v211,v212,v213,v214,v215,v241,v243,v258,v259,v260,v261,v262,v263,v264,v265,v266,v267,v268,v269,v270))

df = subset(df, select = c(satisfied,v98, v224, cntry ,v156, v180, v228, v178, v239, v223, v225 ,v74, v70, v234, v159 ,v122, v227 ,v73, v189, v229, v19, v148, v217, v43, v51, v233
                           ,v56, v71, v186, v242, v102, v41, v253, v226, v84, v139, v216, v184, v218, v179, v232, v173, v147, v183, v135,v85
                           ,v105, v145, v240, v9, v172, v109, v75, v121, v170, v35, v220, v219, v249,v108, v237, v104, v158, v163, v117, v125, v103
                           ,v236, v140,v235,v76,v231,v144,v254,v137,v169,v141,v138,v115,v110,v177,v118,v256,v142,v238,v111,v120,v116,v160,v106,v58,v60,v62,v68,v67,v66,v149,v208,v64,v1,v250,v252,v65,v251,v80,v181,v18,v152,v113,v187,v222,v2,v82,v248,v119,v146,v77,v136,v255,v196,v112,v19,v114,v83,v48,v247,v36,v13,v38,v99,v162,v143,v190,v81,v246,v244,v15,v42,v79,v245,v101,v10))


final = subset(final, select = -c(id,v4,v5,v17,v25,v20,v150,v154,v155,v161,v167,v57,v59,v61,v63,v22,v5,v23,v26,v27,v28,v29,v30,v31,v32,v33,v34, v39, v45,v46,v50,v69,v78,v86,v87,v88,v89,v90,v91,v92,v93,v94,v95,v86,v88,v91,v92,v94,v95,v96,v97,v123,v124,v126, v127, v128, v129, v130, v131, v133, v151, v153,v164,v168,v174,v182,v191,v192,v193,v194,v195,v197,v198,v199,v200,v201,v202,v203,v204,v205,v206,v207,v209,v210,v211,v212,v213,v214,v215,v241,v243,v258,v259,v260,v261,v262,v263,v264,v265,v266,v267,v268,v269,v270))

final = subset(final, select = c(v98, v224, cntry ,v156, v180, v228, v178, v239, v223, v225 ,v74, v70, v234, v159 ,v122, v227 ,v73, v189, v229, v19, v148, v217, v43, v51, v233
                           ,v56, v71, v186, v242, v102, v41, v253, v226, v84, v139, v216, v184, v218, v179, v232, v173, v147, v183, v135,v85
                           ,v105, v145, v240, v9, v172, v109, v75, v121, v170, v35, v220, v219, v249,v108, v237, v104, v158, v163, v117, v125, v103
                           ,v236, v140,v235,v76,v231,v144,v254,v137,v169,v141,v138,v115,v110,v177,v118,v256,v142,v238,v111,v120,v116,v160,v106,v58,v60,v62,v68,v67,v66,v149,v208,v64,v1,v250,v252,v65,v251,v80,v181,v18,v152,v113,v187,v222,v2,v82,v248,v119,v146,v77,v136,v255,v196,v112,v19,v114,v83,v48,v247,v36,v13,v38,v99,v162,v143,v190,v81,v246,v244,v15,v42,v79,v245,v101,v10))


#df <- data.frame(lapply(df, function(x) as.numeric(as.character(x))))
#final <- data.frame(lapply(final, function(x) as.numeric(as.character(x))))


summary(df)


df[] <-  lapply(df, car::recode, 'c(".a",".b",".c",".d")= NA')
final[] <-  lapply(final, car::recode, 'c(".a",".b",".c",".d")= NA')
i1 <- !sapply(df, is.numeric)
Mode <- function(x) { 
  ux <- sort(unique(x))
  ux[which.max(tabulate(match(x, ux)))] 
}
df[i1] <- lapply(df[i1], function(x)
  replace(x, is.na(x), Mode(x[!is.na(x)])))
ok <- sapply(df, is.numeric)
df[ok] <- lapply(df[ok], na.aggregate)


final[] <-  lapply(final, car::recode, 'c(".a",".b",".c",".d")= NA')
i1 <- !sapply(final, is.numeric)
Mode <- function(x) { 
  ux <- sort(unique(x))
  ux[which.max(tabulate(match(x, ux)))] 
}
final[i1] <- lapply(final[i1], function(x)
  replace(x, is.na(x), Mode(x[!is.na(x)])))
ok <- sapply(final, is.numeric)
final[ok] <- lapply(final[ok], na.aggregate)





df$v56 <- car::recode(df$v56,"c('55') = '6'")  
final$v56 <- car::recode(final$v56,"c('55') = '6'")  
df$v58 <- car::recode(df$v56,"c('55') = '6'")  
final$v58 <- car::recode(final$v56,"c('55') = '6'")  
df$v60 <- car::recode(df$v56,"c('55') = '6'")  
final$v60 <- car::recode(final$v56,"c('55') = '6'")  
df$v62<- car::recode(df$v56,"c('55') = '6'")  
final$v62 <- car::recode(final$v56,"c('55') = '6'")
df$v65<- car::recode(df$v56,"c('55') = '8'")  
final$v65 <- car::recode(final$v56,"c('55') = '8'")  
df$v66<- car::recode(df$v56,"c('55') = '8'")  
final$v66 <- car::recode(final$v56,"c('55') = '8'")
df$v67<- car::recode(df$v56,"c('55') = '8'")  
final$v67 <- car::recode(final$v56,"c('55') = '8'")  
df$v68<- car::recode(df$v56,"c('55') = '8'")  
final$v68 <- car::recode(final$v56,"c('55') = '8'")  
df$satisfied <- as.factor(df$satisfied)
#df$v3 <- as.numeric(as.character(df$v3))
#final$v3 <- as.numeric(as.character(final$v3))
#df$v132 <- as.numeric(as.character(df$v132))
#final$v132 <- as.numeric(as.character(final$v132))
df$v250 <- as.numeric(as.character(df$v250))
final$v250 <- as.numeric(as.character(final$v250))
df$v251 <- as.numeric(as.character(df$v251))
final$v251 <- as.numeric(as.character(final$v251))
df$v252 <- as.numeric(as.character(df$v252))
final$v252 <- as.numeric(as.character(final$v252))
#df$v100 <- as.numeric(as.character(df$v100))
#final$v100 <- as.numeric(as.character(final$v100))
df$v64 <- as.numeric(as.character(df$v64))
final$v64 <- as.numeric(as.character(final$v64))



#df <- df %>%
  #mutate_at(vars('v1', 'v2', 'v13', 'v19', 'v56', 'v58', 'v60', 'v62', 'v65', 'v66', 'v67', 'v68', 'v74', 'v75', 'v76', 'v79', 'v80', 'v81', 'v82', 'v83', 'v84', 'v98', 'v99', 'v101', 'v104', 'v105', 'v109', 'v110', 'v111', 'v112', 'v113', 'v114', 'v115', 'v116', 'v117', 'v118', 'v119', 'v120', 'v121', 'v122', 'v135', 'v136', 'v137', 'v138', 'v140', 'v141', 'v142', 'v143', 'v144', 'v145', 'v146', 'v147', 'v148', 'v149', 'v156', 'v177', 'v178', 'v179', 'v180', 'v181', 'v183', 'v184','v186','v189','v219','v220','v222','v223','v224','v225','v226','v227','v232','v233','v234','v235','v236','v237','v238','v239','v240','v249','v250','v251','v252','v253'), as.character)
 
#df <- df %>%
  #mutate_at(vars('v1', 'v2', 'v13', 'v19', 'v56', 'v58', 'v60', 'v62', 'v65', 'v66', 'v67', 'v68', 'v74', 'v75', 'v76', 'v79', 'v80', 'v81', 'v82', 'v83', 'v84', 'v98', 'v99', 'v101', 'v104', 'v105', 'v109', 'v110', 'v111', 'v112', 'v113', 'v114', 'v115', 'v116', 'v117', 'v118', 'v119', 'v120', 'v121', 'v122', 'v135', 'v136', 'v137', 'v138', 'v140', 'v141', 'v142', 'v143', 'v144', 'v145', 'v146', 'v147', 'v148', 'v149', 'v156', 'v177', 'v178', 'v179', 'v180', 'v181', 'v183', 'v184','v186','v189','v219','v220','v222','v223','v224','v225','v226','v227','v232','v233','v234','v235','v236','v237','v238','v239','v240','v249','v250','v251','v252','v253'), as.numeric)

#final <- final %>%
  #mutate_at(vars('v1', 'v2', 'v13', 'v19', 'v56', 'v58', 'v60', 'v62', 'v65', 'v66', 'v67', 'v68', 'v74', 'v75', 'v76', 'v79', 'v80', 'v81', 'v82', 'v83', 'v84', 'v98', 'v99', 'v101', 'v104', 'v105', 'v109', 'v110', 'v111', 'v112', 'v113', 'v114', 'v115', 'v116', 'v117', 'v118', 'v119', 'v120', 'v121', 'v122', 'v135', 'v136', 'v137', 'v138', 'v140', 'v141', 'v142', 'v143', 'v144', 'v145', 'v146', 'v147', 'v148', 'v149', 'v156', 'v177', 'v178', 'v179', 'v180', 'v181', 'v183', 'v184','v186','v189','v219','v220','v222','v223','v224','v225','v226','v227','v232','v233','v234','v235','v236','v237','v238','v239','v240','v249','v250','v251','v252','v253'), as.character)

#final <- final %>%
  #mutate_at(vars('v1', 'v2', 'v13', 'v19', 'v56', 'v58', 'v60', 'v62', 'v65', 'v66', 'v67', 'v68', 'v74', 'v75', 'v76', 'v79', 'v80', 'v81', 'v82', 'v83', 'v84', 'v98', 'v99', 'v101', 'v104', 'v105', 'v109', 'v110', 'v111', 'v112', 'v113', 'v114', 'v115', 'v116', 'v117', 'v118', 'v119', 'v120', 'v121', 'v122', 'v135', 'v136', 'v137', 'v138', 'v140', 'v141', 'v142', 'v143', 'v144', 'v145', 'v146', 'v147', 'v148', 'v149', 'v156', 'v177', 'v178', 'v179', 'v180', 'v181', 'v183', 'v184','v186','v189','v219','v220','v222','v223','v224','v225','v226','v227','v232','v233','v234','v235','v236','v237','v238','v239','v240','v249','v250','v251','v252','v253'), as.numeric)



summary(df)


df1 <- na.omit(df)


df$cntry <- as.numeric(as.factor(df$cntry))
df$cntry <- as.factor(df$cntry)
final$cntry <- as.numeric(as.factor(final$cntry))
final$cntry <- as.factor(final$cntry)






#df$v20 <- as.numeric((df$v20))
#final$v20 <- as.numeric((final$v20))
#df$v20 <- as.factor(df$v20)
#final$v20 <- as.factor(final$v20)


df_x <- df[-1]
final <- rbind(df_x[1, ] , final)
final <- final[-1,]

x <- train[,1:100]
y <- train[,182]
control <- trainControl(search = 'random')
                        
print(rf_default)

df$satisfied <- as.factor(df$satisfied)


df[sapply(df, is.numeric)] <- lapply(df[sapply(df, is.numeric)], as.factor)
#final[sapply(final, is.numeric)] <- lapply(final[sapply(final, is.numeric)], as.factor)
#df[sapply(df, is.numeric)] <- lapply(df[sapply(df, is.numeric)], 
                                       #as.factor)
#df <- na.omit(df)
#df[] <-  lapply(df, car::recode, 
                #'c(".a",".b",".c",".d")=-1')

#final[] <-  lapply(final, car::recode, 
                #'c(".a",".b",".c",".d")=-1')


#df$v223 <- car::recode(df$v223,"c('-1','0','1','2','3','4','5') = '0';  c('6','7','8','9','10')= '1'") 
#df$v224 <- car::recode(df$v224,"c('-1','0','1','2','3','4','5') = '0';  c('6','7','8','9','10')= '1'")  
#df$v225 <- car::recode(df$v225,"c('-1','0','1','2','3','4','5') = '0';  c('6','7','8','9','10')= '1'")  



#df$v223 <- as.numeric(as.character(df$v223))
#df$v224 <- as.numeric(as.character(df$v224))
#df$v225 <- as.numeric(as.character(df$v225))
#df$v226 <- as.numeric(as.character(df$v226))
#df$v227 <- as.numeric(as.character(df$v227))
#df$v98 <- as.numeric(as.character(df$v98))

#df$var = df$v223 * df$v224 * df$v225 * df$v226* df$v227 * df$v98

#df$var  <- with(df,
  ##ifelse(var >= 80000 , 1,0)) 
  
#df$var <- as.factor(df$var)
#df$v223 <- as.factor(df$v223)
#df$v224 <- as.factor(df$v224)
#df$v225 <- as.factor(df$v225)
#df$v226 <- as.factor(df$v226)
#df$v227 <- as.factor(df$v227)
#df$v98 <- as.factor(df$v98)


numrow = nrow(df)
ind = sample(1:numrow,size = as.integer(0.7*numrow))
train <- df[ind, ]
train1 <- df[ind, ]
train2 <-  df[-ind, ]
test <- df[-ind, ]
xtrain <- train[-1]
ytrain <- train[1]
xtest <- test[-176]
ytest <- test[176]
x <- df[-176]

splitSample <- sample(1:3, size=nrow(df), prob=c(0.4,0.4,0.2), replace = TRUE)
train1 <- df[splitSample==1,]
train2 <- df[splitSample==2,]
#test <- df[splitSample==3,]
trainog$satisfied <- as.factor(trainog$satisfied)

calc_class_err = function(actual, predicted) {
  mean(actual != predicted)
}

##Logistic Regression 

modelFitLGtry5 <- speedglm(satisfied ~.,family=binomial(),data = train)
auc(test$satisfied, probtry)
modelFitLG <- glm(satisfied ~v79 + v101 +v98 + v81 + v224 + v225 + v226 + v1 + v74 + v99 + v108  + v217 + v220 + v246 
                    ,family=binomial,data = df)
modelFitLG4 <- speedglm(satisfied ~ v79 + v101 + v98 + v81 + v224 + v225 + v1 + v74 + v99 + v217 + v220 + v246 + v253 + v82 + v223 + v236 + v180 + v233 + v179 +v226 +v178+ v227 + v235 + v237
                  ,family=binomial(),data = train)
stepAIC(modelFitLG4, direction = 'both')
prob4 <- predict(modelFitLG3, final, type="response")
probtry <- predict(modelFitLG4, test, type="response")
prob4 <- as.data.frame(prob4)
pred = rep(0, dim(test)[1])
pred[prob1 > .5] = 1
1-calc_class_err(test$satisfied,pred)

prob2 <- predict(modelFitLG2, final, type="response")
pred = rep(0, dim(test)[1])
pred[prob2 > .5] = 1
1-calc_class_err(test$satisfied,pred)

auc(test$satisfied, probtry)

df$satisfied <- as.factor(df$satisfied)

df$v217 <- as.factor(df$v217)
df$v246 <- as.factor(df$v246)
final$v217 <- as.factor(final$v217)
final$v246 <- as.factor(final$v246)

fit_train <- rpart(satisfied ~ v79 + v101 + v98 + v81 + v224 + v225 + v1 + v74 + v99 + v180 + v217 + v220 + v246 + 
                     v87 + v89 + v93  , data= train, method="class") 
prune_fit_train <- prune(fit_train,cp=fit_train$cptable[which.min(fit_train$cptable[,"xerror"]),"CP"])
pred_test <- predict(prune_fit_train,test) 
predict <- data.frame(pred_test)
predict$pred_value <- ifelse(predict$X1 > predict$X0 , 1, 0)
1-calc_class_err(test$satisfied,predict$pred_value)

x <- subset(train1, select = c(v79, v101 ,v98 , v81 ,v224 , v225 ,v1 ,v74 ,v99 , v180 , v217 , v220 , v246, satisfied) )

tuneRF(df[-201], df$satisfied, 20, ntreeTry=14, stepFactor=2, improve=0.05,
        doBest=TRUE)


control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3, 
                        search='grid')

tunegrid <- expand.grid(.mtry = (1:13)) 

rf_gridsearch <- train(satisfied ~ v79 + v101 +v98 + v81 + v224 + v225 + v226 + v1 + v74 + v99 + v108  + v217 + v220 + v246, 
                       data = df,
                       method = 'rf',
                       metric = 'Accuracy',
                       tuneGrid = tunegrid)
tgrid <- expand.grid(
  .mtry = 1:20,
  .splitrule = "gini",
  .min.node.size = 20
)
x <- subset(train1, select = c(v79 , v101 ,v98 ,v81 ,v224 ,v225 , v1 , v74 , v99 , v217 , v220 , v246 , v253 ,v82 , v223 , v236 , v180 , v233 , v179))
model_caret <- train( y = df2$satisfied,
                      x = df_x,
                     method = "ranger",
                     trControl = trainControl(method="cv", number =2, verboseIter = T, classProbs = T),
                     tuneGrid = tgrid,
                     num.trees = 100,
                     importance = "impurity")
imp <- varImp(model_caret)
varImpPlot(model_caret)

ggplot(v2, aes(x = reorder(names,Overall), y= Overall))+ 
  geom_bar(stat="identity", position="dodge")+ coord_flip()+
  ylab("Variable")+
  xlab("Importance")+
  ggtitle("Variable Importance (Gini)")+
  guides(fill=F)+
  scale_fill_gradient(low="red", high="blue")

v<-as.vector(imp$importance)
v$names <- rownames(v)
v<- v[order(v$Overall, decreasing = TRUE),] 
v2<- v[1:10,]

df2 <- df %>% 
  mutate(satisfied = factor(satisfied, 
                        labels = make.names(levels(satisfied))))
train3$satisfied <- as.factor(train3$satisfied)

summary(df$v58)

df$satisfied <- as.factor(df$satisfied )

##Trees and Random Forest

modelFitRF <- randomForest(satisfied ~. , data=train, mtry =13,ntree = 500)
pred_test <- predict(modelFitRF,test) 
predict <- data.frame(pred_test)
1-calc_class_err(test$satisfied,predict$pred_test)

tuneRF(df[-175], df$satisfied, 13, ntreeTry=100, stepFactor=1.5, improve=1e-5,
       trace=TRUE, plot=TRUE, doBest=TRUE)

tgrid <- expand.grid(
  .mtry = 4:20,
  .splitrule = "gini",
  .min.node.size = 20
)
model_caret <- train(y = train1$satisfied  ~ ., data = train1[-1],
                     method = "ranger",
                     trControl = trainControl(method="cv", number = 3, verboseIter = T, classProbs = F),
                     tuneGrid = tgrid,
                     num.trees = 500,
                     metric = "Accuracy")

rf <- ranger(satisfied ~. , data = train, mtry = 19, num.trees = 500,min.node.size = 20, prob = T)
prob <- predict(rf, final, type="response")
prob5 <- prob$predictions
prob5 <- prob5[,2]
prob5 <- as.data.frame(prob5)
prob10 <- predict(rf, test, type="response")
prob10 <- prob10$predictions
prob10 <- prob10[,2]
prob10 <- as.data.frame(prob10)
auc(test$satisfied,prob10$prob10)

vm <- svm(satisfied ~ ., train1, predict.prob = TRUE)
prob3 <- predict(vm,train2)
train1$satisfied <- as.factor(train1$satisfied )

train1$satisfied


#kNN


knn <- train(
  satisfied ~ v79 + v101 + v98 + v81 + v224 + v225 + v1 + v74 + v99 + v217 + v220 + v246 + v253 + v82 + v236 + v180 + v233 + v179 +v226, 
  data=train12, 
  method='knn',
  tuneGrid=expand.grid(.k=1:20),
  metric='Accuracy',
  trControl=trainControl(
    method='cv', 
    number=2, 
    repeats=2, classProbs =  T))

knn3 <- knn(x, x2, train$satisfied, k = 5, prob = TRUE)
auc(test$satisfied,knn4)
xtrain <- model.matrix(~.,train)
x <- model.matrix(~.,x)
x2 <- model.matrix(~.,x2)
x <- subset(train, select = c(v79, v101 ,v98 , v81 ,v224 , v225 ,v1 ,v74 ,v99 , v180 , v217 , v220 , v246, v253, v236,v233, v226, v179, v82, satisfied) )
x2 <- subset(test, select = c(v79, v101 ,v98 , v81 ,v224 , v225 ,v1 ,v74 ,v99 , v180 , v217 , v220 , v246, v253, v236,v233, v226, v179, v82) )

knn4 <- as.numeric(as.character(knn3))

train$satisfied <- as.numeric(as.character(train$satisfied))

train12 <- train %>% 
  mutate(satisfied = factor(satisfied, 
                        labels = make.names(levels(satisfied))))

train2




#Boosting

modelFitGBM <- gbm(satisfied~., data=train, distribution = 'bernoulli',  n.trees=1000, shrinkage = 0.1)

prob6 <- predict(modelFitGBM ,newdata=final,n.trees=1000,type="response")
prob7 <- predict(modelFitGBM ,newdata=test,n.trees=1000,type="response")
auc(test$satisfied,prob7)
pred = rep(0, dim(test)[1])
pred[prob > .5] = 1
pred <- data.frame(pred)
1-calc_class_err(test$satisfied,pred$pred)
prob2 <- as.data.frame(prob2)

df$satisfied <- as.character(df$satisfied)
trainog$satisfied <- as.character(trainog$satisfied)
train$satisfied <- as.character(train$satisfied)
train1$satisfied <- as.factor(train1$satisfied)
train2$satisfied <- as.factor(train2$satisfied)
summary(train2$satisfied)

training.dt  %>% 
  mutate(churn = factor(churn, 
                        labels = make.names(levels(churn))))


train$satisfied <- as.factor(train$satisfied)
df$satisfied <- as.factor(df$satisfied)
test$satisfied <- as.character(test$satisfied)

v223+v224+v225+v226+v98

 
train1[-175]


xxtrain1 = subset(train1, select = c(v79 , v101 ,v98 , v81 , v224 , v225 , v226 , v1 , v74 , v99 , v108  , v217 , v220 , v246, satisfied))
xxdf = subset(df, select = c(v79 , v101 ,v98 , v81 , v224 , v225 , v226 , v1 , v74 , v99 , v108  , v217 , v220 , v246, satisfied))
xtrain1 <- model.matrix(~.,train1)
xtrain2 <- model.matrix(~.,train2[-1])
xfinal <- model.matrix(~.,final)
xtest <- model.matrix(~.,test)

xtrain <- xtrain[,-1]
xtest <- xtest[,-1]
xfinal <- xfinal[,-1]

xtrain <- make.unique(colnames(xtrain))
make.unique(colnames(xtrain2))
make.unique(colnames(xfinal))

knn <- knn(train =train1[-1] , test = train2[-1],cl = train1$satisfied, prob = TRUE)
#summary(knn)

xtrain22 <- data.frame(xtrain, check.names = FALSE)
names(xtrain22) <- make.unique(names(xtrain22), sep="_")


xtest22 <- data.frame(xtest, check.names = FALSE)
names(xtest22) <- make.unique(names(xtest22), sep="_")

x <- as.numeric(as.character(x))

xtest$satisfied1
xtest <- as.data.frame(xtest)
neural2 <- neuralnet(satisfied ~., data = xtrain22)
prob3 <- predict(neural,xtest)
prob3 <- as.data.frame(prob3)


#Ensembling

modelFitLG23 <- glm(satisfied  ~ prob4 + prob5 +prob6
                  ,family=binomial,data = probi2)

probi1$V1 <- NULL


prob1 <- as.data.frame(prob1)
prob2 <- as.data.frame(prob2) 
prob3 <- as.data.frame(prob3) 
prob4 <- as.data.frame(prob4)
prob5 <- as.data.frame(prob5) 
prob6 <- as.data.frame(prob6) 
probi2 <- bind_cols(prob4, prob5, prob6)
probi2$satisfied <- train2$satisfied

probi1 <- bind_cols(prob1, prob2, prob3)
probi3$predicted <- 0.5*probi3$prob10 + 0.5*probi3$prob6
probi2$predicted <- 0.514*probi2$prob4 + 0.262*probi2$prob5 + 0.224*probi2$prob6
probi1$satisfied <- train2$satisfied
probi3 <- bind_cols(prob10, prob6)
0.027*probi1$prob1 
probtry <-as.data.frame(probtry)
probitry <- bind_cols(probtry, prob10)
probitry$predicted <- 0.65*probitry$probtry + 0.35*probitry$prob10
proc <- auc(test$satisfied, probi3$predicted)
roc(test$satisfied, probi3$predicted)
plot.roc(roc(test$satisfied, probi3$predicted))

predLG <- predict(modelFitLG, val, type="response")
pred = rep(0, dim(val)[1])
pred[predLG > .5] = 1
predLG <- data.frame(pred)
predGBM <- predict(modelFitGBM, newdata = val,n.trees=1000,type="response")
pred = rep(0, dim(val)[1])
pred[predGBM > .5] = 1
predGBM <- data.frame(pred)
prefRF <- predict(modelFitRF, newdata = val)
prefRF <- as.double(as.character(prefRF))
prefRF <- data.frame(prefRF)
predDF <- data.frame(predLG, predGBM, prefRF, satisfied = val$satisfied, stringsAsFactors = F)


modelStack <- train(satisfied ~ ., data = predDF, method = "rf")


predLGtest <- predict(modelFitLG, test, type="response")
pred = rep(0, dim(test)[1])
pred[predLGtest > .5] = 1
predLGtest <- data.frame(pred)
predGBMtest <- predict(modelFitGBM, newdata = test,n.trees=1000,type="response")
pred = rep(0, dim(test)[1])
pred[predGBMtest > .5] = 1
predGBMtest <- data.frame(pred)
prefRFtest <- predict(modelFitRF, newdata = test)
prefRFtest <- as.double(as.character(prefRFtest))
prefRFtest <- data.frame(prefRFtest)
predDFtest <- data.frame(predLGtest, predGBMtest, prefRFtest = unlist(prefRFtest), satisfied = test$satisfied, stringsAsFactors = F)

combPred <- predict(modelStack, predDFtest)


modelStack


fit_train <-rpart(satisfied~v79 + v101 + v98 + v81 + v224 + v225 + v1 + v74 + v99 + v217 + v220 + v246 + v253 + v82 + v236 + v180 + v233 + v179 +v226, data= train, method="class")
prune_fit_train <- prune(fit_train,cp=fit_train$cptable[which.min(fit_train$cptable[,"xerror"]),"CP"])
prob11 <- predict(prune_fit_train,test$satisfied)
prob11 <- as.data.frame(prob11)
auc(test$satisfied, prob11$`0`)




prob1 <- data.frame(prob1)
id <- read.csv('test.csv')
Predicted_final <- data.frame(id$id, probi1$predicted2)
write.csv(Predicted_final,"f21.csv")


v79+v98+v101+v110+v117


knn.cv(train1,cl = train1$satisfied, k = 10, l = 0)




summary(test$satisfied)
summary(test$v64)
df[df=='.a'] <- NA
df[df=='.b'] <- NA
df[df=='.c'] <- NA
df[df=='.d'] <- NA
df <- na.omit(df)df[df==-1] <- NA
df <- na.omit(df)
summary(df$v236)
library(dplyr)
library(rockchalk)


df[204]


summary(logit)
df <- subset( df, select = -id )

df$v3 <- as.numeric(as.character(df$v3))
df$v100 <- as.numeric(df$v100)
df$v132 <- as.numeric(df$v132)
df$v250 <- as.numeric(df$v250)
df$v251 <- as.numeric(df$v251)

logit <- glm(satisfied ~.  ,data = df, family = binomial)
logit.step <- stepAIC(logit, trace = FALSE)
logit.step$anova
library(MASS)

summary(final$v98)
summary(df$satisfied)
summary(df$v98)
1-calc_class_err(df$satisfied,df$v98)


v79+v98+v101+v110+v117+v112+v114+v3




















#LASSO


summary(df$v98)



df_x <- df[,-176]
xdf <- model.matrix(~.,df_x)
xdf <- xdf[,-1]

xtrain1 <- model.matrix(~.,df_x)

ytrain <- df$satisfied
ytrain <- as.data.frame(ytrain)
ytrain <- model.matrix(~.,ytrain) 
#data$Cardio1M=factor(data$Cardio1M)
#split data into train and test
#fitting generalized linear modelalpha=0 then ridge regression is used, while if alpha=1 then the lasso
# of ?? values (the shrinkage coefficient)
#Associated with each value of ?? is a vector of regression coefficients. For example, the 100th value of ??, a very small one, is closer to perform least squares:
Lasso.mod <- glmnet(xdf, ytrain, alpha=1, nlambda=100, lambda.min.ratio=0.0001,family="binomial")
#use 10 fold cross-validation to choose optimal ??.
#cv.out <- cv.glmnet(x, y, alpha=1,family="binomial", nlambda=100, lambda.min.ratio=0.0001,type.measure = "class")
cv.out <- cv.glmnet(xdf, ytrain, alpha=1,family="binomial", nlambda=100, type.measure = "class")
#Ploting the misclassification error and the diferent values of lambda
plot(cv.out)
best.lambda <- cv.out$lambda.min
best.lambda
co<-coef(cv.out, s = "lambda.min")
#Once we have the best lambda, we can use predict to obtain the coefficients.
p <- predict(Lasso.mod, s=best.lambda, newx=xtest, type="class")
inds<-which(co!=0)
variables<-row.names(co)[inds]
variables<-variables[!(variables %in% '(Intercept)')];
variables
variables1
co1 <- sort(co)
print(co)
`%ni%`<-Negate(`%in%`)
inds<-which(co!=0)
variables<-row.names(co)[inds]
variables<-variables[variables %ni% '(Intercept)']

variables

column <- as.data.frame(colnames(df_x))
column

summary(df$v1)











































































































































































































































