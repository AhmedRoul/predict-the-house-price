
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('assignment1_dataset.csv')
le=LabelEncoder()

data['transaction date']=le.fit_transform(data['transaction date'])

#columsX=data.columns
Y=data['house price of unit area']

columsx=['transaction date','house age','distance to the nearest MRT station',
          'number of convenience stores','latitude','longitude']

cls1 = linear_model.LinearRegression()    
X=np.expand_dims(data[columsx[0]], axis=1)
cls1.fit(X,Y) 
prediction= cls1.predict(X)
plt.scatter(X, Y)
plt.xlabel(columsx[0], fontsize = 20)
plt.ylabel('Price', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()
print('Co-efficient of linear regression',cls1.coef_)
print('Intercept of linear regression model',cls1.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y, prediction))

print("-----------------------------")
cls2 = linear_model.LinearRegression()    
X=np.expand_dims(data[columsx[1]], axis=1)
cls2.fit(X,Y) 
prediction= cls2.predict(X)
plt.scatter(X, Y)
plt.xlabel(columsx[1], fontsize = 20)
plt.ylabel('Price', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()
print('Co-efficient of linear regression',cls2.coef_)
print('Intercept of linear regression model',cls2.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y, prediction))

print("-----------------------------")
cls3 = linear_model.LinearRegression()    
X=np.expand_dims(data[columsx[2]], axis=1)
cls3.fit(X,Y) 
prediction= cls3.predict(X)
plt.scatter(X, Y)
plt.xlabel(columsx[2], fontsize = 20)
plt.ylabel('Price', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()
print('Co-efficient of linear regression',cls3.coef_)
print('Intercept of linear regression model',cls3.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y, prediction))

print("-----------------------------")
cls4 = linear_model.LinearRegression()    
X=np.expand_dims(data[columsx[3]], axis=1)
cls4.fit(X,Y) 
prediction= cls4.predict(X)
plt.scatter(X, Y)
plt.xlabel(columsx[3], fontsize = 20)
plt.ylabel('Price', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()
print('Co-efficient of linear regression',cls4.coef_)
print('Intercept of linear regression model',cls4.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y, prediction))

print("-----------------------------")
cls5 = linear_model.LinearRegression()    
X=np.expand_dims(data[columsx[4]], axis=1)
cls5.fit(X,Y) 
prediction= cls2.predict(X)
plt.scatter(X, Y)
plt.xlabel(columsx[4], fontsize = 20)
plt.ylabel('Price', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()
print('Co-efficient of linear regression',cls5.coef_)
print('Intercept of linear regression model',cls5.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y, prediction))

print("-----------------------------")
cls6 = linear_model.LinearRegression()    
X=np.expand_dims(data[columsx[5]], axis=1)
cls6.fit(X,Y) 
prediction= cls6.predict(X)
plt.scatter(X, Y)
plt.xlabel(columsx[5], fontsize = 20)
plt.ylabel('Price', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()
print('Co-efficient of linear regression',cls6.coef_)
print('Intercept of linear regression model',cls6.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y, prediction))


    
# L=.00001
# numberOP=10000

# def model(m0,m1,X):
#     return m0+m1*X

# def predection(m0,m1,X,Y):
#     n=len(X)
#     d0=(1/n)*np.sum(model(m0,m1,X)- Y)
#     d1=(1/n)*np.sum((model(m0,m1,X) -Y)*X)
    
#     m0 =m0- L*d0
#     m1 =m1- L*d1
#     return m0 ,m1

# # to find loss cost in line 
# def optimise(m0,m1,X,Y):
#     for i in range(numberOP):
#         m0,m1=predection(m0,m1,X,Y)
        
        
#     return m0,m1



# Y=data['house price of unit area']
# for Xdata in columsX:
#     if Xdata !='house price of unit area'and Xdata!='distance to the nearest MRT station':
        
#         print("Colums is {}".format(Xdata))
    
#         X=data[Xdata]
    
#         m0=np.random.rand(1)
#         m1=np.random.rand(1)
    
#         m0,m1=optimise(m0,m1,X,Y)
    
#         plt.scatter(X, Y)
#         plt.xlabel(Xdata, fontsize = 20)
#         plt.ylabel('price', fontsize = 20)
    
#         loss=(1/(2*len(X)))*np.sum(np.square(model(m0,m1,X)-Y))
  
    
#         plt.plot(X, model(m0,m1,X), color='red', linewidth = 3)
#         plt.show()
    
#         print("'Mean Square Error' {}".format(loss))
#         print('Co-efficient of linear regression',m1)
#         print('Intercept of linear regression model',m0)
#         print("_____________________________________________________")
#         print("")
 


