import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

file_path = 'C:/Users/User/Downloads/'
file_name = 'car_data.csv'
data = pd.read_csv(file_path + file_name)

data.dropna(inplace=True)
X = data[['Age', 'AnnualSalary']]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

plt.figure(figsize=(25, 20))
plot_tree(dt_model, filled=True,
feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()

feature_importance = dt_model.feature_importances_
plt.bar(X.columns, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

print(f'Training : {accuracy_score(y_train, y_train_pred)}')
print(f'Test : {accuracy_score(y_test, y_test_pred)}')

