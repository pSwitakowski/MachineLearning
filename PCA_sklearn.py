import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import time
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data, target = sklearn.datasets.fetch_openml('mnist_784', return_X_y=True)

data = data[:700]
target = target[:700]

scaler = StandardScaler()

#calculate mean and stadard deviation
scaler.fit(data)
#scale the data
data_scaled = scaler.transform(data)

print('data_scaled shape: ' + str(data_scaled.shape))

pca_090 = PCA(n_components=0.9, random_state=42)
pca_090.fit(data_scaled)
data_pca_090 = pca_090.transform(data_scaled)

print('data_pca_090_var SHAPE: ' + str(data_pca_090.shape))

start_time = time.time()
x_train, x_test, y_train, y_test = train_test_split(data_pca_090, target, test_size=1/7.0, random_state=42)


logisticRegr = LogisticRegression(solver='saga', max_iter=10000)
logisticRegr.fit(x_train, y_train)

y_pred = logisticRegr.predict(x_test)

score = logisticRegr.score(x_test, y_test)

print("Dla 0.90 zachowanej wariancji:")
print('liczba składowych: ' + str(data_pca_090.shape[1]), end='')
print(', Czas: ' + str(time.time() - start_time), end='')
print(", Accuracy: " + str(score*100) + '%')

print('Wariancja zachowana dla poszczególnych parametrów (przy ' + str(data_pca_090.shape[1]) + 'składowych):', np.cumsum(pca_090.explained_variance_ratio_ * 100))

cm = metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)

plt.show()


plt.plot(np.cumsum(pca_090.explained_variance_ratio_))
plt.xlabel("Liczba składowych")
plt.ylabel("Wariancja")
plt.show()