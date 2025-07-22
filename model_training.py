from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = load_iris()
X = df['data']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)

model = LogisticRegression()
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved as model.pkl")





