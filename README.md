# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse

#read csv file
df=
#displaying the content in datafile
df.

#Segregating data to variables
     
#splitting train and test data
     
#import linear regression model and fit the model with the data
     
#displaying predicted values
     
#displaying actual values
     
#graph plot for training data
     
#graph plot for test data
     
#find mae,mse,rmse


## Program:
```
/*
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
.....................................................//(1)
df.tail()
.....................................................//(2)
#segregating data to variables
X=df.iloc[:,:-1].values
X
.....................................................//(3)
Y=df.iloc[:,1].values
Y
.....................................................//(4)
#splitting training and test date
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred
.....................................................//(5)
Y_test
.....................................................//(6)
#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
.....................................................//(7)
#graph plot for test data
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
.....................................................//(8)
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
.....................................................//(9)

Developed by: S>Sajetha
RegisterNumber: 212223100049 

```

## Output:
1.![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/401ddb24-835a-4ecd-8ba5-968e3d40c83c)
2.![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/37f0d000-8d62-4f18-8c48-7907c2770520)
3.![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/7edb8eb1-bd8a-4351-bb7e-ad322e3a8141)
4.![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/9dec2c38-f569-48a8-9b91-bc7de4c729a4)
5.![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/cc0ecb9a-c0e3-422a-90ec-11628a19e98f)
6.![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/ec5ac58a-608e-48a1-941b-37a80f3f96db)
7.![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/9244dc27-eec3-4113-ac3f-05264f91376b)
8.![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/ef583797-cb40-4cdb-bc29-44bbf2cf5f7e)
9.![image](https://github.com/Sajetha13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849316/f0a31892-6647-4b32-b1f6-9199a725a0cc)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
