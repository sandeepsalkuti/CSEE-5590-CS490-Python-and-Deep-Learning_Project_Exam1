import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ---------------------5. a) Data Analysis --------------------------------#
# Loading the data from adult Dataset
adultData = pd.read_csv('adult.csv')
adultData.drop("income",axis=1)
incomeLabel = adultData['income']

# Finding the Values which are missing
print("Number of missing values:\n", format(adultData.isnull().sum()))

# Eliminate NAN values
adultData.dropna(axis = 0, inplace= True)

# Encode the categorial features
trainData = pd.get_dummies(adultData)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainData,incomeLabel)


# ---------------------5. b) Applying Naive Bayes, SVM and KNN--------------------------------#

# ----------- creating the Gaussian Naive Bayes object ---------------#
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# Training the Model
gnb.fit(X_train,y_train)
trainScore = gnb.score(X_train,y_train)

# Predicting the Output
testScore = gnb.score(X_test,y_test)
print(f'\n Gaussian Naive Bayes : Training score - {trainScore} and Test score (Accuracy) - {testScore}')


# ----------- creating the KNN object -------------------------------#
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

# Training the Model
knn.fit(X_train,y_train)
trainScore = knn.score(X_train,y_train)

# Predicting the Output
testScore = knn.score(X_test,y_test)
print(f'\n K Neighbors : Training score - {trainScore} and Test score (Accuracy) - {testScore}')


# ----------- creating the SVM object -------------------------------#

from sklearn import svm
from sklearn.preprocessing import StandardScaler

# Training Model
scaler = StandardScaler()
scaler.fit(trainData,incomeLabel)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Linear SVM Kernel
svc = svm.SVC(kernel='linear')

# Training the Model
svc.fit(X_train_scaled,y_train)
trainScore = svc.score(X_train_scaled,y_train)

# Predicting the Output
testScore = svc.score(X_test_scaled, y_test)

print(f'\n The result of SVM (Linear) is: Training score - {trainScore} and Test score (Accuracy) - {testScore}')


# ---------------------5. c) SVM using Linear and Non-Linear Kernel --------------------------------#

svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = svm.SVC(kernel = kernels[i])
    svc_classifier.fit(X_train_scaled, y_train)
    svc_scores.append(svc_classifier.score(X_test_scaled, y_test))

y_pos = np.arange(len(kernels))
plt.bar(y_pos, svc_scores, color=['black', 'red', 'green', 'blue'])
plt.title('Performance of Linear and Non-Linear SVM Kernel')
plt.xticks(y_pos, kernels)
print(f'\n The result of Non-Linear SVM is: Test score (Accuracy)')
print(f'\n {kernels}')
print(f'\n {svc_scores}')
plt.show()