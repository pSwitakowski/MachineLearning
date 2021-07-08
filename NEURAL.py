import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import time

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def print_sum_num_values(data):
    for column in data:
        print('Null values in column \"', column, '\": ', data[column].isnull().sum())

def print_data_basic_info(data):
    print(data.shape)
    print(data.head())


print("train data before:")
print_data_basic_info(train_data)
print_sum_num_values(train_data)
################################### preparing train data
train_data["Age"] = train_data["Age"].fillna(train_data.Age.mean())
train_data["Embarked"] = train_data["Embarked"].fillna('S') # most common category
train_data.Sex = train_data.Sex.replace(['male', 'female'], [0, 1])
train_data.Embarked = train_data.Embarked.replace(['S', 'C', 'Q'], [0, 1, 2])
train_data['FamilySize'] = train_data['Parch'] + train_data['SibSp']
train_data = train_data.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch', 'SibSp'])
###################################
print("train data after:")
print_data_basic_info(train_data)
print_sum_num_values(train_data)



print("test data before:")
print_data_basic_info(test_data)
print_sum_num_values(test_data)
################################### preparing test data
test_data["Age"] = test_data["Age"].fillna(test_data.Age.mean())
test_data["Fare"] = test_data["Fare"].fillna(test_data.Fare.mean())
test_data.Sex = test_data.Sex.replace(['male', 'female'], [0, 1])
test_data.Embarked = test_data.Embarked.replace(['S', 'C', 'Q'], [0, 1, 2])
test_data['FamilySize'] = test_data['Parch'] + test_data['SibSp']
test_data = test_data.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch', 'SibSp'])
###################################
print("test data after:")
print_data_basic_info(test_data)
print_sum_num_values(test_data)

# train data split
Y_train = train_data['Survived']
X_train = train_data.drop(columns = 'Survived')

#Scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

start_time = time.time()
#solver{‘lbfgs’, ‘sgd’, ‘adam’}
#activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
mlp = MLPClassifier(hidden_layer_sizes=10, activation='identity', solver='lbfgs', random_state=69, max_iter=5000)
mlp.fit(X_train, Y_train)

y_pred = mlp.predict(test_data)

scores = cross_val_score(mlp, X_train, Y_train, cv=15)
end_time = time.time() - start_time

print("Time: " + str(end_time) + ", Accuracy: " + str(scores.mean() * 100) + "%, standard deviation: " + str(scores.std()))

submission = pd.read_csv('gender_submission.csv')
submission["Survived"] = y_pred