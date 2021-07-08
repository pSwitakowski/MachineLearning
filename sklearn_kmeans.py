import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

df = pd.read_csv('Live.csv')

#shape check
print(df.shape)

#summary
print(df.info())

#null values check
print(df.isnull().sum())

#dropping 4 last columns with NaN values
df.drop(columns=['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)

print(df.info())
# 3 non-numerical variables left:
# status_id           7050 non-null object
# status_type         7050 non-null object
# status_published    7050 non-null object

# dropping status_id and status_published. Not dropping status_type as it contains only 4 different values for whole dataset (this column will be our feature column)
df.drop(columns=['status_id', 'status_published'], axis=1, inplace=True)

# features
X = df

# target
y = df['status_type']

# converting text status_type to integers
le = LabelEncoder()
X['status_type'] = le.fit_transform(X['status_type'])
# target variable transformed
y = le.transform(y)

print(X.info())

columns = X.columns
# scaling the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# convert scaler output to pd DataFrame
X = pd.DataFrame(X, columns=columns)

print(X.head())

# k-means alg

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

# printing clusters centers nad inertia value (the lesser the inertia, the better the model is fit)
print("cluster centers:", kmeans.cluster_centers_)
print("inertia:", kmeans.inertia_)

# checking the quality of classification
labels = kmeans.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)

print(correct_labels, " out of ", y.size, " were correctly labeled. Accuracy: ", + correct_labels/float(y.size)*100, "%")

#Using elbow method to find optimal number of clusters
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()