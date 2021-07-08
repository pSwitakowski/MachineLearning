import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import time

data, target = sklearn.datasets.fetch_openml('mnist_784', return_X_y=True)

data = data[:700]
target = target[:700]

start_time = time.time()
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=1/7.0, random_state=42)


logisticRegr = LogisticRegression(solver='saga', max_iter=10000)
logisticRegr.fit(x_train, y_train)

y_pred = logisticRegr.predict(x_test)

score = logisticRegr.score(x_test, y_test)

print('liczba składowych: ' + str(data.shape[1]), end='')
print(', Czas: ' + str(time.time() - start_time), end='')
print(", Accuracy: " + str(score*100) + '%')

cm = metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)

plt.show()
