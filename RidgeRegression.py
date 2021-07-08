import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class RidgeRegression():
    def __init__(self, learning_rate, iterations, l2_penality):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_penality = l2_penality

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
        Y_pred = self.predict(self.X)

        dW = (- (2 * (self.X.T).dot(self.Y - Y_pred)) + (2 * self.l2_penality * self.weight)) / self.training_examples_count
        db = - 2 * np.sum(self.Y - Y_pred) / self.training_examples_count

        self.weight = self.weight - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict(self, X):
        return X.dot(self.weight) + self.b


df = pd.read_csv('salary_data.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.75, random_state=0)

model = RidgeRegression(iterations=1000, learning_rate=0.00001, l2_penality=1)
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
