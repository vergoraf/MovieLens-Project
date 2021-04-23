#########################################################
# Create edx set (training set), validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes.

#Installing packages if it is necessary
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(pander)) install.packages("pander", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)# machine learning procedure
library(data.table)
library (broom)
library(pander)#to create table

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
 #                                          title = as.character(title),
  #                                         genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

save(edx,file="Capstone/edx.rda")
save(validation,file="Capstone/validation.rda")
load(file="Capstone/edx.rda")
load(file="Capstone/validation.rda")


##INTRODUCTION


#Training data dimension
rows<-format(dim(edx)[1],scientific=FALSE)
columns<-dim(edx)[2]
dataframe<-data.frame(rows, columns)
pander(dataframe)

#Dataset example
if(knitr::is_html_output()){
  knitr::kable(head(edx,5), "html") %>%
    kableExtra::kable_styling(bootstrap_options = "striped", full_width = FALSE)
} else{
  knitr::kable(head(edx,5), "latex", booktabs = TRUE) %>%
    kableExtra::kable_styling(font_size = 7)
}

#Unique number of users and movies
users <- n_distinct(edx$userId)
movies <- n_distinct(edx$movieId)
dataframe <- data.frame(users, movies)
pander(dataframe)           # Create the table

#movies classification
# str_detect
genres = c("Drama", "Comedy", "Thriller", "Romance")
pander(sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
}))

##ANALYSIS

#cleaning or preprocessing
edx<-edx%>%select(-timestamp) 


#histogram user and movies
p1 <- edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")
p2 <- edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")
gridExtra::grid.arrange(p2, p1, ncol = 2)

##RESULTS

#choosing lambda
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
# to create 3 training partition to identify the appropiate lambda value
edx_test_index <- createDataPartition(y = edx$rating, times = 3, p = 0.1, list = FALSE)
k<-3
#partition of the training dataset (edx) to apply cross-validation
edx_tr_list<-list()
edx_temp_list<-list()
edx_test_list<-list()
removed<-list()
for (i in 1:k){
edx_tr_list[[i]]<- edx[-edx_test_index[,i],]
edx_temp_list[[i]]<- edx[edx_test_index[,i],]
#make sure
edx_test_list[[i]] <- as.data.frame(edx_temp_list[i]) %>% 
  semi_join(edx_tr_list[[i]], by = "movieId") %>%
  semi_join(edx_tr_list[[i]], by = "userId")
#Add rows
removed[[i]] <- anti_join(as.data.frame(edx_temp_list[i]),as.data.frame(edx_test_list[[i]]))
edx_tr_list[[i]] <- rbind(edx_tr_list[[i]], removed[[i]])
}

#remove temporal unnecessary data
rm(edx_temp_list,removed, edx_temps,control,edx_trainings)

#Procedure is to test  possible lambda values,
#then it is created 3x11 possible RMSE values, using loop, sapply 
#and matrix formulas.

lambdas <- seq(0, 10, 1)# 11 possible lambda values
rmses<-list()
for (j in 1:3) {
rmses[[j]] <- sapply(lambdas, function(l){
  mu <- mean(edx_tr_list[[j]]$rating)
  
  b_i <- edx_tr_list[[j]] %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_tr_list[[j]] %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    edx_test_list[[j]] %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, edx_test_list[[j]]$rating))
})}

#RMSE calculation from the cross-validation step
RMSE_t<-rowMeans(matrix(cbind(rmses[[1]],rmses[[2]],rmses[[3]]),11,3))

qplot(lambdas, RMSE_t) #comparison of lambda values with RMSE
lambda <- lambdas[which.min(RMSE_t)]
lambda# optimal lambda

#Saving values
save(rmses,file="Capstone/rmses.rda")
save(lambda,file="Capstone/lambda.rda")

##Final calculation
mu <- mean(edx$rating) # average of all ratings
b_i <- edx %>% #average ranking for movie i 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u <- edx %>% #user-specific effect u
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

predicted_ratings <- #Prediction 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

RMSE(predicted_ratings, validation$rating) #Final outcome 