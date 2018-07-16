library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)

dataset = read.csv("/Users/Colin/Desktop/cleandata.csv", encoding = 'utf-8')
dataset$Location = as.factor(dataset$Location)
dataset$Towards = as.factor(dataset$Towards)
dataset$Height = as.factor(dataset$Height)
dataset$Decoration = as.factor(dataset$Decoration)

ct <- rpart.control(xval = 10, minsplit = 20, 
                    maxcompete = 6, maxdepth = 30)

fit = rpart(formula = Avg_price ~ .,
                  data = dataset,
                  method = 'anova',
                  parms = list(split = 'gini'),
                  control = ct)
summary(fit)
rpart.plot(fit, branch=0, branch.type=0, type=1,  
           split.cex=1.5, main="Shanghai Second-hand House Pricing",
           extra = 'auto', digits = 5, )

printcp(fit)
plotcp(fit)

ct <- rpart.control(xval = 10, minsplit = 20, cp = 0.003, 
                    maxcompete = 6, maxdepth = 30)
fit = rpart(formula = Avg_price ~ .,
            data = dataset,
            method = 'anova',
            parms = list(split = 'gini'),
            control = ct)
rpart.plot(fit, branch=0, branch.type=0, type=1,  
           split.cex=1.5, main="Shanghai Second-hand House Pricing",
           extra = 'auto', snip = TRUE)
fancyRpartPlot(d)

model2 = randomForest(formula = Avg_price ~ .,data = dataset, proximity=TRUE)
