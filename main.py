import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv("CreditCardBalance.csv")
data['Gender']=data['Gender'].map({' Male':1 ,'Female':0})
data['Student']=data['Student'].map({'Yes':1 ,'No':0})
data['Married']=data['Married'].map({'Yes':1 ,'No':0})
data['Ethnicity']=data['Ethnicity'].map({'Caucasian':1 ,'Asian':2,'African American':3})
X=data.iloc[:,1:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=40)
#Linear Regression
print("# Linear Regression")
linear=LinearRegression()
linear.fit(x_train,y_train)
pred_train_lr=linear.predict(x_train)
pred_test_lr=linear.predict(x_test)
print("mean_squared_error(Train): ",np.sqrt(mean_squared_error(y_train,pred_train_lr)))
print("r2_score(Train): ",r2_score(y_train,pred_train_lr))
print("mean_squared_error(Test): ",np.sqrt(mean_squared_error(y_test,pred_test_lr)))
print("r2_score(Test): ",r2_score(y_test,pred_test_lr))
#Lasso
print("# Lasso Regression")
lasso=Lasso(alpha=5)
lasso.fit(x_train,y_train)
pred_train_lasso=lasso.predict(x_train)
pred_test_lasso=lasso.predict(x_test)
print("mean_squared_error(Train),Lasso: ",np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print("r2_score(Train),Lasso: ",r2_score(y_train,pred_train_lasso))
print("mean_squared_error(Test),Lasso: ",np.sqrt(mean_squared_error(y_test,pred_test_lasso)))
print("r2_score(Test),Lasso: ",r2_score(y_test,pred_test_lasso))
#Ridge

print("# Ridge Regression")
ridge=Ridge(alpha=0.1)
#features_standardized = scaler.fit_transform(x_train)
ridge.fit(x_train,y_train)
pred_train_ridge=ridge.predict(x_train)
pred_test_ridge=ridge.predict(x_test)
print("mean_squared_error(Train),Ridge: ",np.sqrt(mean_squared_error(y_train,pred_train_ridge)))
print("r2_score(Train),Ridge: ",r2_score(y_train,pred_train_ridge))
print("mean_squared_error(Test),Ridge: ",np.sqrt(mean_squared_error(y_test,pred_test_ridge)))
print("r2_score(Test),Ridge: ",r2_score(y_test,pred_test_ridge))

# Elastic Net
print("# Elastic Net")
elastic=ElasticNet(alpha=0.2)
elastic.fit(x_train,y_train)
pred_train_elastic=elastic.predict(x_train)
pred_test_elastic=elastic.predict(x_test)
print("mean_squared_error(Train),Elastic: ",np.sqrt(mean_squared_error(y_train,pred_train_elastic)))
print("r2_score(Train),Elastic: ",r2_score(y_train,pred_train_elastic))
print("mean_squared_error(Test),Elastic: ",np.sqrt(mean_squared_error(y_test,pred_test_elastic)))
print("r2_score(Test),Elastic: ",r2_score(y_test,pred_test_elastic))
