import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
dataset = pd.read_csv('creditcard.csv',sep=',')
dataset.head()
dataset.info() #checking any null val
dataset.isnull().values.any()
## Get the Fraud and the normal dataset
frauddata = dataset[dataset['Class']==1]
normaldata = dataset[dataset['Class']==0]
print(frauddata.shape,normaldata.shape)
frauddata.Amount.describe()
normaldata.Amount.describe()
## Taking only some sample of data as dataset is too large
data= dataset.sample(frac = 0.1,random_state=1)
data.shape
#number of fraud and valid transactions in new dataset
Fd = data[data['Class']==1]
Vd = data[data['Class']==0]
outlier_fraction = len(Fd)/float(len(Vd))
print(Fd.shape,Vd.shape)
print(outlier_fraction)
print("Fraud Cases : {}".format(len(Fd)))
print("Valid Cases : {}".format(len(Vd)))
#independent and Dependent Features
# Define a random state
state = np.random.RandomState(42)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=.2)
##outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(n_estimators=100, max_samples=len(X),
                                        contamination=outlier_fraction, random_state=state, verbose=0),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, algorithm='auto',
                                               leaf_size=30, metric='minkowski',
                                               p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine": OneClassSVM(kernel='rbf', degree=3, nu=outlier_fraction, gamma=0.1,
                                          max_iter=-1)
}
from sklearn.svm import SVC # "Support Vector Classifier"
n_outliers = len(Fd)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != y).sum()
    # Run Classification Metrics
    print("{}: {}".format(clf_name,n_errors))
    print("Accuracy Score :")
    print(accuracy_score(y,y_pred))
    print("Classification Report :")
    print(classification_report(y,y_pred))