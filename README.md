# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step - 1: 
Import the required libaries and read the dataframe.
### Step - 2: 
Assign hours to X and scores to Y.
### Step - 3: 
Implement the training set and the test set of the dataframe.
### Step - 4:
Plot the required graph for both the training data and the test data.
### Step - 5: 
Find the values of MSE,MAE and RMSE.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Santhosh T
RegisterNumber:  212223220100
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("D:/2nd Semester/Intro To ML/marks.csv")
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print("Predicted Y value:",Y_pred)
print("Tested Y value:",Y_test)

plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="navy")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="darkcyan")
plt.plot(X_test,regressor.predict(X_test),color="purple")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)


```


## Output:
![image](https://github.com/SanthoshThiru/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/148958618/c6804303-f8cb-4f38-94e8-5d965f573fc3)
![image](https://github.com/SanthoshThiru/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/148958618/a2623761-3c8d-4923-88c9-510ea5360386)
![image](https://github.com/SanthoshThiru/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/148958618/8bada249-5996-4259-9866-db2001690a9c)
![image](https://github.com/SanthoshThiru/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/148958618/00820ab8-a962-42d8-8d7a-01b2ce984f58)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
