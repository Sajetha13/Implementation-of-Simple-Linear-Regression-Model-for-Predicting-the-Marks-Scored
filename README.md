# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.








## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored
Developed by: S.Sajetha
RegisterNumber: 212223100049
*/
```
```
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()

df.tail()

#segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#splitting training and test date
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:
### head():
![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/4d6d901a-ea54-4917-afb3-4ed97c31005b)
### tail():
![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/0170d304-9c85-46b6-b753-8487f0083a3c)
### X:
![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/5eb52ea6-d0ca-41c0-8b74-776ea629aa6a)
### Y:
![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/cc3abb8d-bd98-4075-b483-08d21f3e878c)
### y_pred:
![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/f5568ca5-b5fe-487b-8b59-d1a9f792f60b)
### y_test:
![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/c89280b6-d92d-4f85-9c1a-adb3841048cd)

![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/241a021f-6edc-47cf-acde-62a11026c2df)
![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/06fe11bb-006b-41af-97c5-0d5984475f10)
![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/ec191dcc-6998-4f4b-8f2d-7dec72fbb19b)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
