import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib import style
style.use('ggplot')

class Kmeans_impl:
    def __init__(self, k=2, tolerance=0.001, max_iter=300):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter

    def fit(self,data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifs = {}

            for i in range(self.k):
                self.classifs[i] = []

            for feature in data:
                distances = [np.linalg.norm(feature-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifs[classification].append(feature)

            prev_centroids = dict(self.centroids)

            for classification in self.classifs:
                self.centroids[classification] = np.average(self.classifs[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tolerance:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


# creating data
X, y = make_blobs(n_samples=50, centers=3, n_features=2, random_state=36)
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()



# using my Kmeans implementation
model = Kmeans_impl(k=3)
model.fit(X)
#new_classification = model.predict(X)
for point in X:
    model.predict(point)


for centroid in model.centroids:
    plt.scatter(model.centroids[centroid][0], model.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in model.classifs:
    color = colors[classification]
    for featureset in model.classifs[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

plt.show()