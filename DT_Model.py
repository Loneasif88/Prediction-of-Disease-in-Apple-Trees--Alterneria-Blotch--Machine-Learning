
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz

from sklearn.model_selection import train_test_split
from sklearn.tree  import DecisionTreeRegressor

df=pd.read_csv("F:/Flask/DataSet/Dataset-90 to 2021.csv")
df1 = df.copy()
df1 = df1.drop('Date',axis=1)
#Normalization on Incidence column
df1['Incd.(%)'].loc[(df1['Incd.(%)'] < 1)] = 0
df1['Incd.(%)'].loc[((df1['Incd.(%)'] >0) & (df1['Incd.(%)'] <= 3.5))] = 1
df1['Incd.(%)'].loc[(df1['Incd.(%)'] >3.5) & (df1['Incd.(%)'] <= 6.5)] = 2
df1['Incd.(%)'].loc[(df1['Incd.(%)'] >6.5) & (df1['Incd.(%)'] <= 12.5)] = 3
df1['Incd.(%)'].loc[(df1['Incd.(%)'] >12.5) & (df1['Incd.(%)'] <= 25.5)] = 4
df1['Incd.(%)'].loc[(df1['Incd.(%)'] >25.5) & (df1['Incd.(%)'] <= 50)] = 5

#Normalization on Severity column
df1['Sevrt.(%)'].loc[(df1['Sevrt.(%)'] < 1)] = 0
df1['Sevrt.(%)'].loc[((df1['Sevrt.(%)'] >0) & (df1['Sevrt.(%)'] <= 3.5))] = 1
df1['Sevrt.(%)'].loc[(df1['Sevrt.(%)'] >3.5) & (df1['Sevrt.(%)'] <= 6.5)] = 2
df1['Sevrt.(%)'].loc[(df1['Sevrt.(%)'] >6.5) & (df1['Sevrt.(%)'] <= 12.5)] = 3
df1['Sevrt.(%)'].loc[(df1['Sevrt.(%)'] >12.5) & (df1['Sevrt.(%)'] <= 25.5)] = 4
df1['Sevrt.(%)'].loc[(df1['Sevrt.(%)'] >25.5) & (df1['Sevrt.(%)'] <= 50)] = 5

#save into new csv File
df1.to_csv("Normalized_Data.csv",index = False)



#slicing
X=df1.iloc[:,0:5]
y = df1.iloc[:,5:]

print(X.shape)
print(y.shape)

# Train testand Split


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state =24)
print("X Training split input -",X_train.shape)
print("X Testing split input -",X_test.shape)
                                
#Decision Tree Algorithm
regressor = DecisionTreeRegressor(max_depth=11, random_state =0)
model = regressor.fit(X_train,y_train)
print("Desicion Tree Classifier created")

#plot Decision Tree Structure
export_graphviz(regressor, out_file = 'tree.dot', feature_names=['Max-Temp','Min-Temp','RH(%)1','RH(%)2','Rain(mm)'])
from graphviz import render
render('dot','png','tree.dot')

#Prediction
y_pred = regressor.predict(X_test)

# Calculate RMSE value
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt

rms = sqrt(mean_squared_error(y_pred,y_test))
print("RMSE value: ",rms)
                                
#calculate R2_square
r2 = r2_score(y_pred,y_test)
print("r2_score :",r2)

#calculate Mean absolute Error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("MAE :",mae)
                                   
pickle.dump(model,open('dt.pkl','wb'))