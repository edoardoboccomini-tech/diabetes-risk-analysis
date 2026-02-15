rm(list=ls())

#Libraries
library(rstudioapi)
library(dplyr)
library(ggplot2)
library(car)
library(psych)
library(cluster)
library(factoextra)
library(caret)
library(glmnet)
library(ResourceSelection)
library(ggcorrplot)
library(ISLR)
library(MASS)
library(pROC)
library(scatterplot3d)

#Environment Configuration
current_path <- getActiveDocumentContext()$path
setwd(dirname(current_path ))
print(getwd())

#Data Loading
df<-readr::read_csv("Diabetes_and_LifeStyle_Dataset.csv")


##--------------------------DATA CLEANING AND PREPROCESSING---------------------------------------------------------

colSums(is.na(df))

df$smoking_status<- factor(df$smoking_status=="Never")
df<-df[df$gender!="Other",]
df$gender<-factor(df$gender=="Male")
df$diagnosed_diabetes<-as.numeric(df$diagnosed_diabetes)
print(df$gender)
print(df$smoking_status)

#Check the structure and duplicates
str(df)
sum(duplicated(df)) 
names(df)
str(df)
summary(df) 
head(df)


##--------------------------DESCRIPTIVE STATISTICS----------------------------------

# Target variable distribution 
total_obs <- nrow(df)
diabetic_individuals <- sum(df$diagnosed_diabetes == 1)
perc_diabetes <- round((diabetic_individuals / total_obs) * 100, 2)
perc_healthy <- 100 - perc_diabetes

tab <- table(df$diagnosed_diabetes)/length(df$diagnosed_diabetes) 
print(tab)

barplot(tab, col=topo.colors(2), xlab=df$diagnosed_diabetes, las=2,
        cex.names=2.5, space=0.5, horiz=TRUE)

glimpse(df)

#BMI boxplot related to diagnosed diabetes
boxplot(bmi~ diagnosed_diabetes, data=df, col=c("lightblue", "pink", 
                                                main="BMI Distribution: Diabetic vs. Non-Diabetic", 
                                                xlab= "Diabetes", ylab="BMI"))

#Boxplot of the main variables
boxplot(df$bmi,
        main="bmi",
        col="lightgreen")

boxplot(df$Age,
        main="Age",
        col="lightpink")

boxplot(df$sleep_hours_per_day,
        main="Sleep hours per day",
        col="lightsalmon")

boxplot(df$physical_activity_minutes_per_week,
        main="Weekly physical activity",
        col="lightgoldenrod")

boxplot(df$diet_score,
        main="Diet score",
        col="purple")

boxplot(df$alcohol_consumption_per_week,
        main="Weekly alcohol consumption",
        col="red")

# Correlation Matrix
sub_df <- df[,c(1,8,9,10,11,13,14,16,18,19,20,21,22,23,24,25,26,27,28,31)]
C=cor(sub_df)
ggcorrplot(C, title= "Correlation Matrix")


##---------------------------LOGISTIC REGRESSION (FULL MODEL)--------------------------------------------- 

logistic_model <- glm(formula=factor(diagnosed_diabetes) ~ Age + 
                        factor(gender) + bmi + 
                        physical_activity_minutes_per_week + sleep_hours_per_day +
                        factor(smoking_status) + diet_score +
                        factor(hypertension_history) + factor(cardiovascular_history) +
                        alcohol_consumption_per_week + factor(family_history_diabetes),
                      data = df, family = binomial(link = "logit"))

summary(logistic_model)

# Goodness of Fit: MCFADDEN R^2
logistic_model$null.deviance 
logistic_model$deviance
mf_r2 <- 1-logistic_model$deviance/logistic_model$null.deviance
print(mf_r2)

# LIKELIHOOD RATIO TEST (LRT) 
LR <- logistic_model$null.deviance - logistic_model$deviance
k <- length(logistic_model$coefficients)-1 
print(k)
p_value_LR <- pchisq(LR, k,lower.tail=FALSE)
print(p_value_LR) 

# Multicollinearity Check: VIF
vif(logistic_model)

phat <- logistic_model$fitted.values 
#xb <- predict(logistic_model) 
summary(phat) 

# Prediction 
yhat <- as.numeric(phat>0.5)  
summary(as.factor(yhat))
summary(as.factor(df$diagnosed_diabetes))

y_true <- as.numeric(as.character(df$diagnosed_diabetes))
print(y_true)

# Metrics for predictions
# Confusion Matrix
c0 <- sum((yhat==0 & y_true==0)) 
c1 <- sum((yhat==1 & y_true==1)) 
w1 <- sum((yhat==0 & y_true==1)) 
w0 <- sum((yhat==1 & y_true==0)) 

tab_predictions <- rbind(cbind(c1,w1),cbind(w0,c0)) #the Confusion Matrix, built combining c0,c1,w0,w1
colnames(tab_predictions) <- c("Predicted sick","Predicted healthy")
rownames(tab_predictions) <- c("Real sick","Real healthy")
print(tab_predictions)

#Accuracy
accuracy <- (c0 + c1) / length(df$diagnosed_diabetes) #=Percentuale di previsioni corrette totali.
print(accuracy)

#Sensitivity
sensitivity <- c1/(c1+w1) #=Ability to correctly identify real positives
print(sensitivity)

#Specificity
specificity <- c0/(c0+w0) #=Ability to correctly identify real negatives
print(specificity) #in this case is not so good (it is a conditional probability)

# ROC curve and AUC
roc <- roc( df$diagnosed_diabetes ~ logistic_model$fitted.values) #it is a curve
plot(roc, xlab='1-specificity',ylab='sensitivity' )    
auc(roc) 

## -------------------------MODEL OPTIMIZATION (STEPWISE) ------------------------------------------------

model_log_final= step(logistic_model, direction="both")
summary(model_log_final)

# Goodness of Fit: MCFADDEN R^2
model_log_final$null.deviance
model_log_final$deviance 

mf_r2_1 = 1-model_log_final$deviance/model_log_final$null.deviance
print(mf_r2_1)

phat_1 <- logistic_model$fitted.values

#AUC and ROC curve
roc_1 <- roc( df$diagnosed_diabetes ~ model_log_final$fitted.values) 
plot(roc_1, xlab='1-specificity',ylab='sensitivity' )
auc(roc_1) 


# Search for the Best Threshold
best_coord <- coords(roc_1, "best",
                     ret=c("threshold","accuracy",
                           "sensitivity", "specificity"))
                         
print(best_coord)

yhat_1 <- as.numeric(phat_1 > 0.5922172)

# Final metrics
# Confusion matrix
c0_1 = sum((yhat_1==0 & y_true==0))  
c1_1 = sum((yhat_1==1 & y_true==1))  
w1_1 = sum((yhat_1==0 & y_true==1)) 
w0_1 = sum((yhat_1==1 & y_true==0)) 

tab_predictions_1 <- rbind(cbind(c1_1,w1_1),cbind(w0_1,c0_1)) 
colnames(tab_predictions_1)<-c("Predicted sick","Predicted healthy")
rownames(tab_predictions_1)<-c("Real sick","Real healthy")
print(tab_predictions_1)

#Accuracy
accuracy_1 = (c0_1 + c1_1) / length(df$diagnosed_diabetes) 
print(accuracy_1)

#Sensitivity
sensitivity_1=c1_1/(c1_1+w1_1) 
print(sensitivity_1)

#Specificity
specificity_1=c0_1/(c0_1+w0_1) 
print(specificity_1) 

# Average Partial Effects (APE)
betalogit=model_log_final$coefficients
xb_1 <- predict(model_log_final) 
dens_hat <- exp(xb_1)/((1+exp(xb_1))^2) 
beta_predictors <- betalogit[-1] 
PE <- dens_hat %*% t(beta_predictors) 
head(PE)

APE = colMeans(PE) # Average Partial Effect = the mean of effects for each predictor/column
print(APE)


##------------------------- PRINCIPAL COMPONENT ANALYSIS (PCA) ----------------------------------------------------------------------

# Standardization
# Identification of numeric columns to standardize: excluding binary variables (0/1) and the Logistic target variable
vars_to_scale<- c("bmi", "Age", "physical_activity_minutes_per_week",
                         "diet_score")
pca_df <- df[, vars_to_scale] 
summary(pca_df)
df_scaled <- scale(pca_df)
colMeans(df_scaled) # check that means are approximately 0
apply(df_scaled, 2, sd) # check that standard deviations are approximately 1

# PCA Execution
pca_result <- princomp(df_scaled, cor=TRUE, scores=TRUE)
summary(pca_result) 
str(pca_result)
print(summary(pca_result),loading=TRUE)

# Scree Plot
fviz_eig(pca_result, 
         choice = "variance",       
         addlabels = TRUE,          
         barfill = "lightblue",       
         barcolor = "lightblue",      
         linecolor = "black",      
         ncp = 4,                   
         ggtheme = theme_minimal()  
) + labs(title = "Scree plot",
       x = "Dimensions",
       y = "Percentage of explained variances")

#3D Plot
pca_result$scores
scatterplot3d(pca_result$scores[,1],  
              pca_result$scores[,2],
              pca_result$scores[,3],
              pch = 19,
              color = "blue",
              xlab = " PC1 (Obesity Risk)",
              ylab = "PC2 (Sedentary Lifestyle)",
              zlab = "PC3")



##-------------------------- LINEAR DISCRIMINANT ANALYSIS (LDA) ------------------------------------------------------

# Creation of standardized dataset
vars_factors <- c("cardiovascular_history", "family_history_diabetes", "diagnosed_diabetes")
vars_total <- c(vars_to_scale, vars_factors)

lda_df <- df[,vars_total]
lda_df[, vars_to_scale] <- scale(lda_df[, vars_to_scale])

lda_df$diagnosed_diabetes <- factor(lda_df$diagnosed_diabetes)
lda_df$cardiovascular_history <- factor(lda_df$cardiovascular_history)
lda_df$family_history_diabetes <- factor(lda_df$family_history_diabetes)

#Model Estimation and Results
lda_model <- lda(diagnosed_diabetes ~ ., data = lda_df) 
lda_model

lda_model$counts 
lda_model$means 
lda_model$scaling 
lda_model$lev

# Predictions
lda_pred <- predict(lda_model) 
scores <- lda_pred$x
View(scores)
pred_class= lda_pred$class  

# Density Plot
lda_plot_data <- data.frame(
  LD1_Score = scores,
  Diagnosis = lda_df$diagnosed_diabetes)

ggplot(lda_plot_data, aes(x = LD1, fill = Diagnosis)) +
  geom_density(alpha = 0.6) +
  labs(title = "Distribution of Discriminant Scores (LD1)",
       x = "LD1 Score",
       y = "Density") +
  theme_minimal() 

# Confusion Matrix
conf_matrix <- confusionMatrix(pred_class, lda_df$diagnosed_diabetes, positive = "1") 
conf_matrix

#AUC and ROC curve
post_probs <- lda_pred$posterior[,1] #This is the precise probability that a patient belongs to group 0 or group 1.
roc_lda <- roc( lda_df$diagnosed_diabetes ~ post_probs ) #it is a curve
plot(roc_lda,
     main="ROC curve: LDA model",
     xlab='1-Specificity',
     ylab='Sensitivity')

auc_lda <- auc(roc_lda)
auc_lda


##-------------------- QUADRATIC DISCRIMINANT ANALYSIS (QDA) -------------------------------------------

qda_model <- qda(diagnosed_diabetes ~ ., data = lda_df)
print(qda_model)
qda_model$counts  

# Predictions
qda_pred <- predict(qda_model) 
pred_class_qda <- qda_pred$class 
post_probs_qda <- qda_pred$posterior[, "1"] 

# QDA Confusion Matrix
conf_mat_qda <- confusionMatrix(pred_class_qda, lda_df$diagnosed_diabetes, positive = "1")
print(conf_mat_qda)

# ROC curve and AUC 
roc_qda <- roc( lda_df$diagnosed_diabetes, post_probs_qda) # Creates ROC curve
auc_qda <- auc(roc_qda)
print(auc_qda)

## ----------------- MODEL COMPARISON (LDA vs QDA)--------------------------------------------------------------------------------------

# Visual ROC Comparison
plot(roc_lda, 
     main="ROC Comparison: LDA vs QDA",
     xlab='1-Specificity',
     ylab='Sensitivity')
lines(roc_qda, col="blue")
legend("bottomright", 
       legend=c("LDA", "QDA"), 
       col=c("black", "blue"), 
       lwd=2)

# Correlation Calculation
print(colnames(lda_pred$posterior))
print(colnames(qda_pred$posterior))

prob_lda <- lda_pred$posterior[, "1"] # extracting column 1 for both
prob_qda<- qda_pred$posterior[, "1"]
correlation <- cor(prob_lda, prob_qda)
print(paste("Correlation LDA vs QDA:", round(correlation, 4)))

# Scatterplot Comparison
comparison_df <- data.frame(
  Prob_LDA = prob_lda,
  Prob_QDA = prob_qda,
  Real_diagnosis = lda_df$diagnosed_diabetes)
comparison_df

ggplot(comparison_df, aes(x = Prob_LDA, y = Prob_QDA, color = Real_diagnosis)) +
  geom_point(alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +  #Line y=x
  labs(title = "Posterior Probability Comparison: LDA vs QDA",
       subtitle = paste("Correlation:", round(correlation, 3)),
       x = "Diabetes Probability (LDA)",
       y = "Diabetes Probability (QDA)") +
  theme_minimal() +
  scale_color_manual(values = c("0" = "green", "1" = "red"))



##-------------------------------- CLUSTER ANALYSIS -------------------------------------------------------------

# Hierarchical Clustering (for k determination)
vars_lifestyle <- c("Age", "bmi", "physical_activity_minutes_per_week", "diet_score")
df_cluster_input <- df[, vars_lifestyle]
df_scaled <- scale(df_cluster_input)

# Dendrogram on sample of 1000
set.seed(123)
sample_idx <- sample(1:nrow(df_scaled), 1000)
dist_matrix <- dist(df_scaled[sample_idx, ], method = "euclidean")
dendro <- hclust(dist_matrix, method = "ward.D2")
plot(dendro, labels = FALSE, main = "Dendrogram (Sample)", xlab = "", sub = "")

# Scree Plot of Fusions
last_height <- tail(dendro$height, 20)

plot(20:1, last_height,
     type = "b",
     main = "Scree Plot (Last 20 Fusions)",
     xlab = "Number of Clusters (k)",
     ylab = "Fusion Height (Distance)",
     col = "blue", pch = 19)


# K-Means Clustering
set.seed(123)
km_final <- kmeans(df_scaled, centers = 4, nstart = 25)
df$Cluster_Lifestyle <- as.factor(km_final$cluster)
df$Cluster_Lifestyle

# Cluster Profiling
cluster_profile <- df %>%
  group_by(Cluster_Lifestyle) %>%
  summarise(
    How_many = n(),
    Avg_Age = mean(Age),
    Avg_bmi = mean(bmi),
    Avg_Physical_Activity= mean(physical_activity_minutes_per_week),
    Avg_Diet = mean(diet_score),
    Perc_Family_History = mean(as.numeric(as.character(family_history_diabetes))) * 100,
    Perc_Diabetic = mean(as.numeric(as.character(diagnosed_diabetes))) * 100)

View(cluster_profile)


# Cluster Visualization
fviz_cluster(list(data = df_scaled, cluster = km_final$cluster),
             geom = "point",        
             ellipse.type = "convex", 
             palette = "jco",       
             ggtheme = theme_minimal(),
             main = "Visualization of 4 Lifestyle Clusters")

