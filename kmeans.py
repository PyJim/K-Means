import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.cluster import KMeans

cars = pd.read_csv("cars.csv")
#for i in cars.columns:
    #cars[i]=cars[i].fillna("0")
X = cars[cars.columns[:-1]]
def to_float(cell):
    try:
        return float(cell)
    except:
        return 0
for column in X.columns:
    X[column]=X[column].apply(to_float)

for col in X.columns:
    mean = X[col].mean()
    X[col]=X[col].replace(0,mean)
    #X[col]=X[col].astype(np.float).astype("Int32")


wcss = []
for i in range(0,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300,n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(0,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300,n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(X)
X=X.as_matrix(columns=None)

plt.scatter(X[y_kmeans== 0,0],X[y_kmeans==0,1], s=100,c="red",label="Toyota")
plt.scatter(X[y_kmeans== 1,0],X[y_kmeans==1,1], s=100,c="blue",label="Nissan")
plt.scatter(X[y_kmeans== 2,0],X[y_kmeans==2,1], s=100,c="green",label="Honda")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=100,c="yellow",label="Centroids")
plt.title("Clusters of Car Make")
plt.legend()
plt.show()
