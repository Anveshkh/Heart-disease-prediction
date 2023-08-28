
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.preprocessing import StandardScaler
np.random.seed(42)

pd.set_option("display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)  # Prevent line wrapping

heart_disease = pd.read_csv("heart-disease.csv")
print(heart_disease.head())

# Distribution of Age
plt.hist(heart_disease["age"])
plt.title("Distribution of Age", color="green")
plt.xlabel("Age", color="red")
plt.ylabel("Frequency", color="red")
# plt.show()

# Distribution of Sex
plt.hist(heart_disease["sex"])
plt.title("Distribution of Sex", color="green")
plt.xlabel("Sex", color="red")
plt.ylabel("Frequency", color="red")
# plt.show()

# Distribution of cp
cp_vc = heart_disease["cp"].value_counts()
plt.barh(["0", "1", "2", "3"], cp_vc)
plt.title("Distribution of cp", color="green")
plt.xlabel("Frequency", color="red")
plt.ylabel("cp Categories", color="red")
# plt.show()

# Distribution of Cholestrol
plt.hist(heart_disease["chol"])
plt.title("Distribution of Chol", color="green")
plt.xlabel("Cholesterol", color="red")
plt.ylabel("Frequency", color="red")
# plt.show()

# Distribution of trestbps
plt.hist(heart_disease["trestbps"])
plt.title("Distribution of trestbps", color="green")
plt.xlabel("trestbps", color="red")
plt.ylabel("frequency", color="red")
# plt.show()

# Distribution of target
plt.hist(heart_disease["target"])
print(heart_disease["target"].value_counts())
plt.title("Distribution of Target")
plt.xlabel("Target", color="red")
plt.ylabel("frequency", color="red")
# plt.show()

# Relationship between sex and target
rel1 = heart_disease.groupby("target")["sex"].value_counts().unstack()
rel1.plot(kind="bar")
plt.title("Relationship between Sex and Target")
plt.xlabel("Target and SEX", color="red")
plt.ylabel("Frequency", color="red")
# plt.show()

# Relationship between cp and target
rel2 = heart_disease.groupby("target")["cp"].value_counts().unstack()
rel2.plot(kind="bar")
plt.title("Relationship between cp and target")
plt.ylabel("Frequency", color="red")
# plt.show()

# Relationship between age and target
plt.scatter(heart_disease["age"], heart_disease["target"])
plt.title("Relationship between age and  target")
plt.xlabel("Age", color="red")
plt.ylabel("Target", color="red")
# plt.show()

# Relationship between trestbps and target
plt.scatter(heart_disease["trestbps"], heart_disease["target"])
plt.title("Relationship between target and trestbps")
plt.xlabel("Trestbps")
plt.ylabel("target")
# plt.show()

# Calculate the shape of the dataset
rows, cols = heart_disease.shape

# Check for missing values
print(heart_disease.isna().sum())

# Checking the dtype of all columns
print(heart_disease.dtypes)

# Making X and Y datasets
X = heart_disease.drop("target", axis=1)
Y = heart_disease["target"]

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Instantiating a Logistic Regression model
model = LogisticRegression(n_jobs=-1, max_iter=1000)

# Fitting the model to the training data
model.fit(X_train, y_train)

# Evaluating the model
lr_score = model.score(X_test, y_test)
lr_y_preds = model.predict(X_test)
lr_acc_score = accuracy_score(y_test, lr_y_preds)
lr_precision_sc = precision_score(y_test, lr_y_preds)
lr_class_report = classification_report(y_test, lr_y_preds)
print("Model 1")
print("Logistic Regression Score = ", round(lr_score, 2))
print("Logistic Regression Accuracy score = ", round(lr_acc_score, 2))
print("Logistic Regression Precision score = ", round(lr_precision_sc, 2))
print("Logistic Regression Classification report = \n", lr_class_report)


# model 2 :
model2 = RandomForestClassifier(n_estimators=100)
model2.fit(X_train, y_train)
rf_score = model2.score(X_test, y_test)
rf_y_preds = model2.predict(X_test)
rf_acc_score = accuracy_score(y_test, rf_y_preds)
rf_precision_sc = precision_score(y_test, rf_y_preds)
rf_class_report = classification_report(y_test, rf_y_preds)
print("Model 2")
print("Random Forest Score = ", round(rf_score, 2))
print("Random Forest Accuracy score ", round(rf_acc_score, 2))
print("Random Forest Precision score", round(rf_precision_sc, 2))
print("Random Forest Classification report\n", rf_class_report, 2)
print("")

# model 3 :
model3 = GaussianNB()
model3.fit(X_train, y_train)
gnb_score = model3.score(X_test, y_test)
gnb_y_preds = model3.predict(X_test)
gnb_acc_score = accuracy_score(y_test, gnb_y_preds)
gnb_precision_sc = precision_score(y_test, gnb_y_preds)
gnb_class_report = classification_report(y_test, gnb_y_preds)
print("Model 3")
print("Naive Bayes score = ", round(gnb_score, 2))
print("Naive Bayes Accuracy score = ", round(gnb_acc_score, 2))
print("Naive Bayes Precision score = ", round(gnb_precision_sc, 2))
print("Naive Bayes Classification report = \n", gnb_class_report)

## Scaling the data in the range of 0-1

scaler = StandardScaler()
scaled_df = scaler.fit_transform(heart_disease.drop("target", axis=1))
cols = X_train.columns
new_df = pd.DataFrame(scaled_df, columns=cols)

new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(new_df, Y, test_size=0.2)

model.fit(new_X_train, new_y_train)

print("Performance after Standardization = ", model.score(new_X_test, new_y_test))