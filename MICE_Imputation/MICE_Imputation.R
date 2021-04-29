trainSet <- read.table("./trainSet/trainSet.txt", header=T, sep=",", na.strings=c("NaN","<undefined>"))
testSet <- read.table("./testSet/testSet.txt", header=T, sep=",", na.strings=c("NaN","<undefined>"))
totalSet <- rbind(trainSet, testSet)
totalSet$Cough <- as.factor(totalSet$Cough)
totalSet$DifficultyInBreathing <- as.factor(totalSet$DifficultyInBreathing)
totalSet$CardiovascularDisease <- as.factor(totalSet$CardiovascularDisease)
totalSet$RespiratoryFailure <- as.factor(totalSet$RespiratoryFailure)
totalSet$Prognosis <- as.factor(totalSet$Prognosis)

library(finalfit)
library(dplyr)
library(mice)

######################MCAR/MAR_Analysis#######################
# --> Conclusion: MAR

NAs <- sapply(trainSet, function(x) {sum(is.na(x))})
total <- rep(nrow(trainSet), length(NAs))
cbind(NAs, total)  

trainSet %>%
  missing_plot()
testSet %>%
  missing_plot()

trainSet %>% 
  select(-PatientID) %>% 
  select(-Hospital) %>% 
  select(-ImageFile) %>% 
  ff_glimpse()

explanatory = c("Age", "Sex", 
                "Temp_C",  
                "Cough", "DifficultyInBreathing", "WBC", "CRP", 
                "Fibrinogen", 
                "LDH", "Ddimer", "Ox_percentage", "PaO2",
                "SaO2", "pH", "CardiovascularDisease", "RespiratoryFailure")
dependent = "Prognosis"

trainSet %>% 
  select(-PatientID) %>% 
  select(-Hospital) %>% 
  select(-ImageFile) %>% 
  missing_pattern(dependent, explanatory)

trainSet %>% 
  select(-PatientID) %>% 
  select(-Hospital) %>% 
  select(-ImageFile) %>% 
  summary_factorlist(dependent, explanatory, 
                     na_include=TRUE, p=TRUE)

trainSet %>%
  select(-PatientID) %>% 
  select(-Hospital) %>% 
  select(-ImageFile) %>% 
  missing_pairs(dependent, explanatory)


##########################################Imputation with MICE##############################

trainSet %>% 
  missing_predictorMatrix(
    #drop_from_imputed = c("Prognosis", "ImageFile"),
    drop_from_imputed = "ImageFile",
    drop_from_imputer = c("Hospital", "PatientID", "ImageFile")
  ) -> predM

tempData <- mice(totalSet, predictorMatrix=predM, m=1, maxit=100, meth="pmm")
summary(tempData)
completedData <- complete(tempData, 1)
#xyplot(tempData, pH ~ Ox_percentage,pch=18,cex=1)
densityplot(tempData)

#manually impute ImageFile values:
completedData$ImageFile <- sapply(completedData$PatientID,function(x){paste(x,".png", sep="")})

completedTrainSet <- completedData[1:nrow(trainSet),]
completedTestSet <- completedData[(nrow(trainSet)+1):nrow(completedData),]

#round to digits according original test set
completedTestSet$Age <- sapply(completedTestSet$Age,round, digits=0)
completedTestSet$SaO2 <- sapply(completedTestSet$SaO2,round, digits=0)
completedTestSet$PaO2 <- sapply(completedTestSet$PaO2,round, digits=0)
completedTestSet$LDH <- sapply(completedTestSet$LDH,round, digits=0)

completedTestSet$pH <- sapply(completedTestSet$pH,round, digits=2)

write.table(completedTrainSet, file="./trainSet/trainSet_Imputed.txt", sep=",", quote = FALSE, row.names=F)
write.table(completedTestSet, file="./testSet/testSet_Imputed.txt", sep=",", quote = FALSE, row.names=F)

sum(is.na(completedTrainSet))
sum(is.na(completedTestSet))

imputeMICE <- function(dataset, methods) {
  tempDataframe <- mice(dataset, predictorMatrix=predM, m=1, maxit=10, method=methods)
  return(completedDataframe <- complete(tempDataframe, 1))
}

#######Evaluation#########

sigmas <- c(
  #"Age": 
  15.05,
  #"Temp_C"
  0.97,
  #"WBC"
  3.53,
  #"CRP"
  66.93,
  #"Fibrinogen"
  158.36,
  #"LDH"
  235.22,
  #"Ddimer"
  6743.13,
  #"Ox_percentage"
  7,
  #"PaO2"
  26.11,
  #"SaO2"
  8.24,
  #"pH"
  0.06,
  #"Sex"
  1,
  #"Cough"
  1,
  #"DifficultyInBreathing"
  1,
  #"CardiovascularDisease"
  1,
  #"RespiratoryFailure"
  1
)
type <- c("str", "str", "str", "num", "str", "num", "str", "str", rep("num", 9), "str", "str", "str")

filenames <- list.files("/Users/marco.stock/Documents/COVID/HIDA_COVID_Alpha_X_hackathon/data/train_set_with_missing_vals", pattern="*.txt", full.names=TRUE)
ldf <- lapply(filenames, read.table, header=T, sep=",", na.strings=c("NaN","<undefined>"))

evaluate <- function(groundTruthc, imputedc, missingc, sigma, typec) {
  m1<- rep(2, length(groundTruthc))
  if(typec=="str") {
    m1 <- sapply((((as.integer(as.logical(groundTruthc == imputedc))) / sigma)^2 ), min, 1)
  }
  else if(typec=="num"){
    m1 <- sapply((((groundTruthc - imputedc) / sigma)^2 ), min, 1)
  }
  #print(m1)
  #print(missingc)
  #print(m1[missingc])
  print(mean((m1[missingc])))
  return(mean(m1[missingc]))
}

ImputeandEval <- function(methods) {
  mean <- rep(2, length(ldf))
  for(i in 1:length(ldf)) {
    toImpute <- ldf[[i]]
    #print(filenames[[i]])
    imputed <- imputeMICE(toImpute, methods)
    #write.table(imputed, file=paste(filenames[[i]],"_MICE_RF.txt"), sep=",", quote = FALSE, row.names=F)
    metrics <- rep(2, length(sigmas))
    for(c in 1:length(sigmas)) {
      #print(paste("Spalte", colnames(groundTruth[c])))
      metrics[c] <- evaluate(groundTruth[,(c+3)], imputed[,(c+3)], (is.na(toImpute[,(c+3)])&!is.na(groundTruth[,(c+3)])), sigmas[c], type[(c+3)])
    }
    mean[i] <- mean(metrics)
    #print(mean[i])
  }
  return(mean)
}

groundTruth <- totalSet[(1:nrow(trainSet)),]
meth_cont <- "ri"
meth_cat <- "pmm"
#methods <- list(c(rep(meth_cat, 3),meth_cont,meth_cat,meth_cont,rep(meth_cat,2),rep(meth_cont, 9), rep(meth_cat,3)))
methods <- list(c(rep("pmm",3),"midastouch", "pmm", "mean", "norm.nob","ri","mean", "mean", "pmm", "midastouch", "pmm", "cart","mean","cart","mean","cart","logreg","logreg"))
sapply(methods, ImputeandEval)


