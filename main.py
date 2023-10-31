import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Step 1: Load the dataset and remove the "DiabetesPedigreeFunction" column
data = pd.read_csv("diabetes.csv")


# Step 2: Split the dataset into training and testing sets
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Apply Logistic Regression and KNN
lr_model = LogisticRegression(max_iter=1000)
knn_model = KNeighborsClassifier()

lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# Step 5: Make predictions and evaluate the models
lr_pred = lr_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)

print("Logistic Regression Accuracy: {:.2f}%".format(lr_accuracy * 100))
print("KNN Accuracy: {:.2f}%".format(knn_accuracy * 100))

# You can use other classification metrics as well, e.g., precision, recall, F1-score, ROC-AUC.
# Use classification_report for a detailed report.

print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_pred))

print("KNN Classification Report:")
print(classification_report(y_test, knn_pred))

# Step 6: Hyperparameter tuning using GridSearchCV
# Define parameter grids for each algorithm
lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
knn_param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

# Perform GridSearchCV for each model
lr_grid_search = GridSearchCV(lr_model, lr_param_grid, cv=5)
knn_grid_search = GridSearchCV(knn_model, knn_param_grid, cv=5)

# Fit the models with the best hyperparameters
lr_grid_search.fit(X_train, y_train)
knn_grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_lr_params = lr_grid_search.best_params_
best_knn_params = knn_grid_search.best_params_

print("Best Logistic Regression Hyperparameters:", best_lr_params)
print("Best KNN Hyperparameters:", best_knn_params)

