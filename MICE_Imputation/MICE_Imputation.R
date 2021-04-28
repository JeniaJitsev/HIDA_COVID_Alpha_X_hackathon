trainSet <- read.table("./trainSet/trainSet.txt", header=T, sep=",", na.strings=c("NaN","<undefined>"))
testSet <- read.table("./testSet/testSet.txt", header=T, sep=",", na.strings=c("NaN","<undefined>"))
totalSet <- rbind(trainSet, testSet)

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
xyplot(tempData, pH ~ Ox_percentage,pch=18,cex=1)
densityplot(tempData)
stripplot(tempData, pch = 20, cex = 1.2)

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
