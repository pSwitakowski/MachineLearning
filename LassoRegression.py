import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LassoRegression():
    def __init__(self, learning_rate, iterations, l1_penality):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penality = l1_penality

    def fit(self, X, Y):
        self.training_examples_count, self.features_count = X.shape

        self.weight = np.zeros(self.features_count)

        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):
        predicted_Y = self.predict(self.X)

        delta_W = np.zeros(self.features_count)

        for j in range(self.features_count):
            if self.weight[j] > 0:
                delta_W[j] = (-(2 * (self.X[:, j]).dot(self.Y - predicted_Y)) + self.l1_penality) / self.training_examples_count
            else:
                delta_W[j] = (-(2 * (self.X[:, j]).dot(self.Y - predicted_Y)) - self.l1_penality) / self.training_examples_count

        delta_b = -2 * np.sum(self.Y - predicted_Y) / self.training_examples_count

        self.weight = self.weight - self.learning_rate * delta_W
        self.b = self.b - self.learning_rate * delta_b
        return self

    def predict(self, X):
        return X.dot(self.weight) + self.b


df = pd.read_csv('salary_data.csv')

X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

model = LassoRegression(iterations=1000, learning_rate=0.0001, l1_penality=1)
model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)
print("Oszacowane wartości: ", np.round(Y_pred[:5], 2))
print("Wartości rzeczywiste: ", Y_test[:5])
print("waga: ", round(model.weight[0], 2))
print("b: ", round(model.b, 2))

plt.scatter(X_test, Y_test, color='cyan')
plt.plot(X_test, Y_pred, color='magenta')
plt.title('Salary - Years of experience')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()
