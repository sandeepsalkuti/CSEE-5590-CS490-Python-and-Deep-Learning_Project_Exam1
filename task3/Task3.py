import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
data=pd.read_csv("weather.csv")          #Loading dataset
data.head()
data.shape
data.isnull().any()  #checking for null values in all columns
data['Precip Type'].value_counts()

data.loc[data['Precip Type'].isnull(),'Precip Type']='rain'     #replacing null value in column with rain
data.isnull().any()

data.loc[data['Precip Type']=='rain','Precip Type']=1
data.loc[data['Precip Type']=='snow','Precip Type']=0
data.drop(['Summary', 'Daily Summary','Formatted Date'], axis=1, inplace=True)

X=data.drop(['Temperature (C)'],axis=1)       #taking X and y values which is dependent and independent features
y=data['Temperature (C)']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1)  #splitting training and testing data
#linear regression model
model=LinearRegression()
model.fit(X_train,y_train)
pred=model.predict(X_test)
print(np.mean((pred-y_test)**2))   #calculating error using linear regression
pd.DataFrame({'actualvalue':y_test,
              'predictedvalue':pred,
              'difference':(y_test-pred)})

#polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)            #calculating error using polynomial
tran=poly.fit_transform(X_train)
poly.fit(tran,y_train)
model=LinearRegression()
model.fit(tran,y_train)
pred=model.predict(poly.fit_transform(X_test))
print(np.mean((pred-y_test)**2))
pd.DataFrame({'actualvalue':y_test,
              'predictedvalue':pred,
              'difference':(y_test-pred)})

#Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor
reg1=RandomForestRegressor(max_depth=50,random_state=42,n_estimators=100)
reg1.fit(X_train,y_train)
pred1=reg1.predict(X_test)
print(np.mean(pred1-y_test)**2)            #calculating error rate using mean value of predicted minus test value
pd.DataFrame({'actualvalue':y_test,
              'predictedvalue':pred1,
              'difference':(y_test-pred1)})

accuracy = reg1.score(X_train,y_train)  #As randomforest yields lesser errorrate using it as our model
print(accuracy)

accuracy = reg1.score(X_test,y_test)
print(accuracy)
