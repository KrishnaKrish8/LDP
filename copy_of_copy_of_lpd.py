import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
import pickle

"""/content/indian_liver_patient.csv"""

data = pd.read_csv("indian_liver_patient.csv")
data = data.dropna()
data = data.drop('Gender', axis=1)
X = data.drop('Dataset', axis=1)
Y = data['Dataset']
rf = RandomForestClassifier()
rf.fit(X, Y)
importances = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
importances = importances.sort_values('importance', ascending=False)
print(importances)
a = importances['feature'].values
for i in range(1, 4):
    X = X.drop(a[-i], axis=1)
#X = X.drop('Albumin', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X.shape)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=39)
#pickle.dump(scaler, open("scaler.pkl", "wb"))
"""In KNN"""

print('KNN:')
model1 = KNeighborsClassifier(n_neighbors=18)
model1.fit(x_train, y_train)
y_pred1 = model1.predict(x_test)
accuracy = accuracy_score(y_pred1, y_test) * 100
print(accuracy)
precision = precision_score(y_test, y_pred1) * 100
print("Precision: {:.2f}".format(precision))
#pickle.dump(model1, open("knnmodel.pkl", "wb"))

"""AdaBoost"""

print('AdaBoost:')
dmodel = DecisionTreeClassifier(criterion='entropy', max_depth=1)
model2 = AdaBoostClassifier(estimator=dmodel, n_estimators=43, learning_rate=0.7)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
# print(y_pred2)
accuracy = accuracy_score(y_pred2, y_test) * 100
print("Accuracy:", accuracy)
precision = precision_score(y_test, y_pred2) * 100
print("Precision: {:.2f}".format(precision))
#pickle.dump(model2, open("adamodel.pkl", "wb"))

"""ANN"""

model3 = Sequential()
model3.add(Dense(6, input_dim=6, activation='tanh'))
model3.add(Dense(3, activation='tanh'))
#model3.add(Dense(4, activation='tanh'))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.fit(x_train, y_train, epochs=100, batch_size=10)
y_pred3 = model3.predict(x_test)
print(y_pred3)
y_pred3 = (y_pred3 == float(1))
y_pred3 = [1 if x == True else 2 for x in y_pred3]
print('ANN:')
accuracy = accuracy_score(y_pred3, y_test) * 100
print(accuracy)
precision = precision_score(y_test, y_pred3) * 100
print("Precision: {:.2f}".format(precision))
#pickle.dump(model3, open("annmodel.pkl", "wb"))

"""Majority Voting"""


def hifr(x, y, z):
    a = x
    if y == a:
        return a
    elif z == a:
        return a
    else:
        return y


ans = [hifr(y_pred1[i], y_pred2[i], y_pred3[i]) for i in range(len(x_test))]
print(ans)
accuracy = accuracy_score(ans, y_test) * 100
print(accuracy)
precision = precision_score(ans, y_test) * 100
print("Precision: {:.2f}".format(precision))
