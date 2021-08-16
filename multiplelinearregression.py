import pandas as pd

dataset = pd.read_csv("insurance.csv")
print(dataset.shape)

dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
dataset.replace({'region': {'southwest': 0, 'northwest': 1, 'southeast': 2, 'northeast': 3}}, inplace=True)

X = dataset.drop(columns='charges')
y = dataset['charges']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

prediction_train=model.predict(X_train)
prediction_test=model.predict(X_test)

from sklearn.metrics import r2_score
optimization_train=r2_score(y_train,prediction_train)
optimization_test=r2_score(y_test,prediction_test)
print("R2 Score of Training Data :" ,optimization_train)
print("R2 Score of Testing Data :" ,optimization_test)

import pickle #saving the model
with open('modelRegression.pkl', 'wb') as file:
    pickle.dump(model, file)