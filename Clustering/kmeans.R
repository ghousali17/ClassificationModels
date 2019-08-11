# K-Means Clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
dataset = dataset[4:5]

set.seed(123)
#Use elbow method to find optimal number of clusters.

wcss <- vector()
#within column returns wcss value for each cluster
for(i in 1:10) wcss[i] <- sum(kmeans(dataset,i)$withins)

#type='b' tells the plot function that we need a line graph
plot(1:10, wcss, type='b',main = paste('Cluster of clients'),
     xlab = 'Number of clusters',
     ylab = 'WCSS value')

#Fit the kmeans algorithm to our dataset
kmeans <- kmeans(dataset,5)

#install.packages('cluster')


library(cluster)
clusplot(dataset,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         labels = 2,
         plotchar= FALSE,
         span = TRUE,
         main = paste('Cluster of Clients'),
         xlab = 'Annual Income',
         ylab = 'Spending score')
