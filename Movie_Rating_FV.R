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
if(!require(rlist)) install.packages("rlist", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)#pipeline production and data management
library(caret)# machine learning procedure
library(data.table)
library (broom)
library(pander)#to create table
library(rlist)# more functions for list management
library(lubridate)#to manage time data

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

save(edx,file="Capstone/rda_data/edx.rda")
save(validation,file="Capstone/rda_data/validation.rda")
load(file="Capstone/rda_data/edx.rda")
load(file="Capstone/rda_data/validation.rda")


##INTRODUCTION

##ANALYSIS

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

#histogram user and movies
p1 <- edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  labs(x='number of ratings',y="number of movies")+ 
  scale_x_log10() + 
  ggtitle("Movies")
p2 <- edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  labs(x='number of ratings',y="number of users")+
  scale_x_log10() + 
  ggtitle("Users")
gridExtra::grid.arrange(p2, p1, ncol = 2)
#users concentrated around 80 ratings per user and a movie being evaluated with
#100 ratings

#histogram users and movies according to rating
p2<-edx%>%group_by(userId)%>% summarize(avg=mean(rating))%>% 
  ggplot(aes(avg)) + geom_histogram(bins=30, color='black') +
  labs(x='user average ratings',y='rating counts') 
p1<-edx%>%group_by(movieId)%>% summarize(avg=mean(rating))%>% 
  ggplot(aes(avg)) + geom_histogram(bins=30, color='black')+
  labs(x='Movie average ratings',y='rating counts') 
gridExtra::grid.arrange(p2, p1, ncol = 1)

##movies genre classification
#How many genres are in the dataset?
sep_genre<-list.append(str_split(edx$genres,'\\|'))
genres_edx<-unique(combine(sep_genre))
save(genres_edx,file="Capstone/rda_data/genres_edx.rda")
load(file="Capstone/rda_data/genres_edx.rda")
rm(sep_genre)

#How many ratings are for each genre?
genres_count<-sapply(genres_edx, function(g) {
  sum(str_detect(edx$genres, g))})
genres_count<-as.data.frame(stack(genres_count))%>% arrange(desc(.))#stack allows to manage named vectors
colnames(genres_count) <- c("count","genre")
pander(genres_count)# Create the table
save(genres_count, file="Capstone/rda_data/genres_count.rda")

#How is the rating for each genre? are there a difference among them?
#Identifying average rating
#table average, se (adapted from dfedeoli calculations)
genre_rating<-matrix(nrow=20, ncol=3)#matrix with ave and se values
j<-1 #fundamental for looping
for (i in genres_edx){
  grating<-edx[which(str_detect(edx$genres,i)),]%>% 
    summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n()))#calculating value 
  genre_rating[j,1]<-grating$avg#avg value for each genre
  genre_rating[j,2]<-grating$se# se value for each genre
  genre_rating[j,3]<-i# genre label
  j<-j+1
}
#genre graph
#it is filtered "no files" genre category, due to high Standard Error (0.4)
genre_rating<-as.data.frame(genre_rating)
save(genre_rating,file="Capstone/rda_data/genre_rating.rda")
load(file="Capstone/rda_data/genre_rating.rda")

colnames(genre_rating)<- c("avg","se","genre")
ggplot(filter(genre_rating,se<=0.1), aes(y=reorder(genre,as.numeric(avg),FUN=mean), x=as.numeric(avg))) + 
  geom_point(size=1)+ 
  geom_errorbarh(aes(xmin=as.numeric(avg)-2*as.numeric(se),
                     xmax = as.numeric(avg)+2*as.numeric(se)),
                 height=0.3, colour="black", alpha=0.9, size=0.5)+
  theme_minimal()+
  labs(y = "" ) +
  labs(x = "Average ranking") 
#it is observable a genre influence

#Rating date
rat_time<-edx%>%mutate(daterat=as_datetime(timestamp))%>%
  select(rating,daterat,genres)
rat_time$ym<-round_date(rat_time$daterat,unit="month")
rat_time$season<- quarter(rat_time$daterat)
save(rat_time,file="Capstone/rda_data/rat_time.rda")
#boxplot
rat_time%>%
  group_by(ym)%>%
  summarise(avg=mean(rating))%>%
  ggplot(aes(y=avg))+
  geom_boxplot()
#practically all data is below 3.8

#graph of 4 seasons analysing rating date
rat_time%>%
  group_by(ym,season)%>%
  mutate(ym=ym, season=factor(recode(season,"1"="Winter","2"="Spring","3"="Summer","4"="Autumn"),
                       levels=c("Spring","Summer","Autumn","Winter")))%>%
  summarise(avg=mean(rating))%>%arrange()%>%
  filter(avg<=3.8)%>%
  ggplot(aes(ym,avg))+
  geom_point()+labs(y="average rating",x="Year")+
  geom_smooth(formula= 'y~x',method="loess", se=T)+
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5,size=7),legend.position="top")+
  ylim(3,4)+scale_x_datetime(date_breaks = "1 year",date_labels="%Y")+
  scale_color_identity()+facet_grid(.~season)
#no significant effect of rating date in the dataset.
#therefore decided to use movie, user and genre effect.

#Genre
#Here it will be calculated just the media effect without penalizations since almost
#all the genre have enough n to calibrate the result
#plus the extraordinary memory consumption

#Calculating ßK (effect of each genre)

#Due to hardware processing ( it was impossible to apply at 1 shot)
#1.- start  subdividing the dataset according to each genre
edx_<-list()
for (i in 1:length(genres_edx)){
  a<-as.data.frame(sapply(genres_edx[i], function(x) {
    str_detect(edx$genres, x)}))
  edx<-cbind(edx,a)
  edx_[[i]]<-edx%>%filter(!!sym(genres_edx[i])== TRUE)
  edx<-select(edx,-7)
}
rm(a)
save(edx_,file="Capstone/rda_data/edx_.rda")
load(file="Capstone/rda_data/edx_.rda")

#2.- establish b_k for each genre
mu <- mean(edx$rating)
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_i - mu))

b_k.list<-list()
for(i in 1:length(genres_edx)){
b_k <- edx_[[i]] %>% 
  left_join(b_u, by="userId") %>%
  left_join(b_i, by="movieId")%>%
  summarize(b_k = mean(rating - b_i - b_u - mu))
b_k.list[[i]]<-b_k
}
save(b_k.list,file="Capstone/rda_data/b_k.list.rda")
load(file="Capstone/rda_data/b_k.list.rda")
rm(b_i,b_k,b_u,mu,edx_)

#3.- Calculation sum beta_k for each rate
#is done in this way to not overwhelm memory processing
m_kvalues<-matrix(unlist(b_k.list))
#subdivide edx in 10 times
#subbdivision
edx_1<-edx[1:1000000,]
#matrix value genre
a_1<-as.matrix(sapply(genres_edx, function(x) {
  ifelse(str_detect(edx_1$genres, x)=='TRUE',1,0)}))
#sum of  each rate  with genre b_k values
k_1<-a_1%*%m_kvalues
rm(edx_1,a_1)
#subbdivision
edx_2<-edx[1000001:2000000,]
#matrix value genre
a_2<-as.matrix(sapply(genres_edx, function(x) {
  ifelse(str_detect(edx_2$genres, x)=='TRUE',1,0)}))
#sum of  each rate  with genre b_k values
k_2<-a_2%*%m_kvalues
rm(edx_2,a_2)
#subbdivision
edx_3<-edx[2000001:3000000,]
#matrix value genre
a_3<-as.matrix(sapply(genres_edx, function(x) {
  ifelse(str_detect(edx_3$genres, x)=='TRUE',1,0)}))
#sum of  each rate  with genre b_k values
k_3<-a_3%*%m_kvalues
rm(edx_3,a_3)
#subbdivision
edx_4<-edx[3000001:4000000,]
#matrix value genre
a_4<-as.matrix(sapply(genres_edx, function(x) {
  ifelse(str_detect(edx_4$genres, x)=='TRUE',1,0)}))
#sum of  each rate  with genre b_k values
k_4<-a_4%*%m_kvalues
rm(edx_4,a_4)
#subbdivision
edx_5<-edx[4000001:5000000,]
#matrix value genre
a_5<-as.matrix(sapply(genres_edx, function(x) {
  ifelse(str_detect(edx_5$genres, x)=='TRUE',1,0)}))
#sum of  each rate  with genre b_k values
k_5<-a_5%*%m_kvalues
rm(edx_5,a_5)
#subbdivision
edx_6<-edx[5000001:6000000,]
#matrix value genre
a_6<-as.matrix(sapply(genres_edx, function(x) {
  ifelse(str_detect(edx_6$genres, x)=='TRUE',1,0)}))
#sum of  each rate  with genre b_k values
k_6<-a_6%*%m_kvalues
rm(edx_6,a_6)
#subbdivision
edx_7<-edx[6000001:7000000,]
#matrix value genre
a_7<-as.matrix(sapply(genres_edx, function(x) {
  ifelse(str_detect(edx_7$genres, x)=='TRUE',1,0)}))
#sum of  each rate  with genre b_k values
k_7<-a_7%*%m_kvalues
rm(edx_7,a_7)
#subbdivision
edx_8<-edx[7000001:8000000,]
#matrix value genre
a_8<-as.matrix(sapply(genres_edx, function(x) {
  ifelse(str_detect(edx_8$genres, x)=='TRUE',1,0)}))
#sum of  each rate  with genre b_k values
k_8<-a_8%*%m_kvalues
rm(edx_8,a_8)
#subbdivision
edx_9<-edx[8000001:9000000,]
#matrix value genre
a_9<-as.matrix(sapply(genres_edx, function(x) {
  ifelse(str_detect(edx_9$genres, x)=='TRUE',1,0)}))
#sum of  each rate  with genre b_k values
k_9<-a_9%*%m_kvalues
rm(edx_9,a_9)
#subbdivision
edx_10<-edx[9000001:9000055,]
#matrix value genre
a_10<-as.matrix(sapply(genres_edx, function(x) {
  ifelse(str_detect(edx_10$genres, x)=='TRUE',1,0)}))
#sum of  each rate  with genre b_k values
k_10<-a_10%*%m_kvalues
rm(edx_10,a_10)
# append k values 
k<-rbind(k_1,k_2,k_3,k_4,k_5,k_6,k_7,k_8,k_9,k_10)
rm(k_1,k_2,k_3,k_4,k_5,k_6,k_7,k_8,k_9,k_10)
#add k value to edx
edx$b_k<-k
rm(k)


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
rm(edx_temp_list,removed)

#Procedure is to test  possible lambda values,
#then it is created 3x11 possible RMSE values, using loop, sapply 
#and matrix formulas.

lambdas <- seq(0, 10, 0.5)# 21 possible lambda values
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
    mutate(pred = mu + b_i + b_u + b_k) %>%# b_k it was calculated as a mean value in the previous step
    pull(pred)
  
  return(RMSE(predicted_ratings, edx_test_list[[j]]$rating))
})}

#RMSE calculation from the cross-validation step
RMSE_t<-rowMeans(matrix(cbind(rmses[[1]],rmses[[2]],rmses[[3]]),length(lambdas),3))

qplot(lambdas, RMSE_t) #comparison of lambda values with RMSE
lambda <- lambdas[which.min(RMSE_t)]
lambda# optimal lambda

#Saving values
save(rmses,file="Capstone/rda_data/rmses.rda")
save(lambda,file="Capstone/rda_data/lambda.rda")

#delete unnecessary data, due to memory processing
rm(edx_test_index,edx_test_list,edx_tr_list)

##Final calculation
mu <- mean(edx$rating) # average of all ratings
b_i <- edx %>% #average ranking for movie i 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u <- edx %>% #user-specific effect u
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

#b_k
m_kvalues<-matrix(unlist(b_k.list))
#matrix value genre
val_k<-as.matrix(sapply(genres_edx, function(x) {
  ifelse(str_detect(validation$genres, x)=='TRUE',1,0)}))
#sum of  each rate  with genre b_k values
val_k<-val_k%*%m_kvalues
#add k value to edx
validation$b_k<-val_k
rm(val_k)

predicted_ratings <- #Prediction 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u+b_k) %>%# b_k it was calculated as a mean value in the previous step
  pull(pred)

RMSE(predicted_ratings, validation$rating) #Final outcome 
