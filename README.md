# unsupervised-machine-learning-model

In machine learning we have two types of  machine learning models
!- Supervised ml models-where we have a target column 
2-Unsupervised  ml models - where we dont  have a target column( The machine learns and identfies patterns and grops based  on the similar patterns

### Project Overview
In the following project we will be clustering mall customers into groups and clusters.The number of clusters is also identified  by the model ,we dont jsut assume the number of clusters we will group our data into,Grouping of these customers maybe on several features(maybe the frequency in which they visit the mall,the amount of money they spend in the mall,
the number of items they purchase in the mall and so much more)

In the following we will  be using the **KMeans** clustering model.

## Data ingestion and cleaning
 we will be ingesting the documnet file of mall customers and also look for null values (remember if less attenion is given in determining the datatypes and cleaning the data then the model will give you challenges later while trying to build it.
 ```
#we willl start by importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
os.environ["THREADPOOLCTL_THREAD_COUNT"] = "1"
df =pd.read_csv("Mall_Customers.csv")
```


### EDA 
 We will look athe the shape and info of our data y using the below commands
 `df.dscribe()`
 
 `df.shape`
 
 `df.info`

 We will look the the scores distributionof our data  and  create the scores and look the amount of people with different **spending scores**
 ```
ss_1_20 =df['Spending Score (1-100)'][(df['Spending Score (1-100)']>=1)& (df['Spending Score (1-100)']<=20)]
ss_21_40 =df['Spending Score (1-100)'][(df['Spending Score (1-100)']>=21)& (df['Spending Score (1-100)']<=40)]
ss_41_60 =df['Spending Score (1-100)'][(df['Spending Score (1-100)']>=41)& (df['Spending Score (1-100)']<=60)]
ss_61_80 =df['Spending Score (1-100)'][(df['Spending Score (1-100)']>=61)& (df['Spending Score (1-100)']<=80)]
ss_81_100 =df['Spending Score (1-100)'][(df['Spending Score (1-100)']>=81)& (df['Spending Score (1-100)']<=100)]

ssx =['1-20','21-40','41-60','61-80','81-100']
ssy=[len(ss_1_20.values),len(ss_21_40.values),len(ss_41_60.values),len(ss_61_80.values),len(ss_81_100.values)]
plt.figure(figsize=(15,6))
sns.barplot(x=ssx,y=ssy,palette ='rocket')
plt.title('spending scores')
plt.xlabel('score')
plt.ylabel('number of customers having the score')
plt.show()

```
![a](https://github.com/stilinsk/unsupervised-machine-learning-model/assets/113185012/7c7b3094-a960-404f-be47-38816a7b8964)

As we can clearly see above we have the most customers with a spending score of 41-60 and least from 21-40.we also have a significant amount of customers from 811-100

We will also look at the distribution of the **annual income**
```

ai0_30=df['Annual Income (k$)'][(df['Annual Income (k$)']>=0)&(df['Annual Income (k$)']<=30)]
ai31_60=df['Annual Income (k$)'][(df['Annual Income (k$)']>=31)&(df['Annual Income (k$)']<=60)]
ai61_90=df['Annual Income (k$)'][(df['Annual Income (k$)']>=61)&(df['Annual Income (k$)']<=90)]
ai91_120=df['Annual Income (k$)'][(df['Annual Income (k$)']>=91)&(df['Annual Income (k$)']<=120)]
ai121_150=df['Annual Income (k$)'][(df['Annual Income (k$)']>=121)&(df['Annual Income (k$)']<=150)]


ssx =['1-30k$','31-60k$','61-90k$','91-120k$','121-150k$']
ssy =[len(ai0_30.values),len(ai31_60.values),len(ai61_90.values),len(ai91_120.values),len(ai121_150)]
plt.figure(figsize=(15,6))
sns.barplot(x=ssx,y=ssy,palette ='rocket')
plt.title('annual income')
plt.xlabel('annual income')
plt.ylabel('number of customers having the annual income')
plt.show()
```
![a](https://github.com/stilinsk/unsupervised-machine-learning-model/assets/113185012/5301d80a-e60f-4f89-9e51-4422597d832c)

In the above barchat we can clearly see the annual most annual income eaners earn between 61-90k

In the following line of code we will be plotting and elbow like features ,where the elbow is steepest thats where we have the  reccommended amount of clusers

'''    it is creating a KMeans object with a varying number of clusters, ranging from 1 to 10, and 
computing the within-cluster sum of squares (WCSS) for each clustering solution. The WCSS is a measureof how compact
the clusters are, and it is commonly used to determine the optimal number of clusters in k-means clustering.
The for loop iterates through the range of cluster numbers from 1 to 10, and for each iteration, it creates a KMeans object 
with the specified number of clusters (n_clusters=i) and an initialization method of "k-means++" (init='k-means++'). 
The fit method is then called on the KMeans object to fit the model to the data in X. The inertia_ attribute of the KMeans 
object is then appended to the wcss list. The inertia_ attribute gives the WCSS value for the current clustering solution.'''

```

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```

'''   These values can be plotted to create an "elbow plot" that shows the trade-off between the number 
of clusters and the compactness of the resulting clusters. The "elbow" of the plot indicates the point of diminishing
returns in terms of increased cluster compactness, and it is often used to determine the optimal number of clusters for a
given dataset.'''

```
plt.figure(figsize =(6,6))
plt.grid()
plt.plot(range(1,11),wcss,linewidth =2,color ='red',marker ='8')
plt.xlabel('kvalue')
plt.ylabel('wcss')
plt.show()
```
![a](https://github.com/stilinsk/unsupervised-machine-learning-model/assets/113185012/522d8285-58ce-42f0-989f-7eecbb55962c)

for the above plot as its clearly shown the elbow is steepest at oround 5 that is where we have an ebow forming at that region so as clearly shown by the code we clearly see the optimal numberof clusters to use is five clusters

 *For the below lines of code we specify that the clusters should be 5 and we also specify we are using the k++ ALGORITHM*
*and then we are fittin the model and we are assigning it to a varibale y*
```
kmeans =KMeans(n_clusters =5 ,init ='k-means++',random_state =0)
#return a label for each datapoint based on thei cluster
Y=kmeans.fit_predict(X)
Y
```

The scatter plot is created using the plt.scatter function from the matplotlib library. The function is called multiple times to plot the data points in each cluster, with different colors used for each cluster. The X[Y==0,0] syntax is used to select the data points in the first cluster, where Y==0 is a Boolean array that is True for all data points in the first cluster and False for all other data points. The first column of X corresponds to the "Annual Income" feature, so X[Y==0,0] selects the "Annual Income" values for the data points in the first cluster. Similarly, X[Y==0,1] selects the "Spending Score" values for the data points in the first cluster.

The s=50 parameter sets the size of the markers in the scatter plot, while the c parameter specifies the color of the markers for each cluster. The label parameter is used to provide a label for each cluster, which is used in the legend that is created using the plt.legend function.
```
plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='blue', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='orange', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='purple', label='Cluster 5')
plt.legend()
plt .scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c ='cyan',label='')
plt.title('Customer groups')
plt.xlabel('Annnual income')
plt.ylabel('Spending score')

plt.show()
```
![a](https://github.com/stilinsk/unsupervised-machine-learning-model/assets/113185012/285285fd-2866-4c54-871b-905682bc909c)




