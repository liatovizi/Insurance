import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

data = pd.read_csv("insurance.csv")
print(data.describe())
print(data.info())
print(data.head())

data.region.value_counts()
#null values
print(data.isna().sum())

#dependent,independent variables
y = data.charges
x= data.drop(['charges'], axis=1)
print(y.head())
print(x.head())

#encodeyes no to 1,0
x.smoker = x.smoker.eq('yes').mul(1)
x.sex = x.sex.eq('female').mul(1)
#one_hot encoding
cat_var = list(x.select_dtypes(include=["object"]))

print(cat_var)

x = pd.get_dummies(x, columns=cat_var, drop_first = True)

#outliners
columns = list(x)
print(columns)
warnings.filterwarnings("ignore", category=plt.cbook.mplDeprecation)

sns.set(rc={'figure.figsize': (20, 10)})

for j in range(1, 9):
    plt.subplot(2, 4, j)
    sns.boxplot(x=x[columns[j - 1]])


z = np.abs(stats.zscore(x))
print(z)
threshold = 3
print(np.where(z > 3))

#Statmodel Regression
#OLS statsmodel regression
from statsmodels.stats import diagnostic

import statsmodels.api as sm
model = sm.OLS(y, x).fit()
#predictions = model.predict(x_test)

print_model = model.summary()
print(print_model)

#Sklearn regression
#train_test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
#Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict
y_pred = regressor.predict(x_test)

#Accuracy
from sklearn.metrics import r2_score
r2_test = r2_score(y_test, y_pred)

from sklearn.metrics import mean_squared_error
from math import sqrt
mse_test = sqrt(mean_squared_error(y_test, y_pred))
print(r2_test)
print(mse_test)

y_predt = regressor.predict(x_train)
r2_train = r2_score(y_train, y_predt)
mse_train = sqrt(mean_squared_error(y_train, y_predt))
print(r2_train)
print(mse_train)

#find the residuals
residual = y_test - y_pred

#Testing assumptions
plt.scatter(y_pred,y_test)
sns.set(rc={'figure.figsize':(20,10)})

for j in range(1, 9):
    plt.subplot(2, 4, j)
    plt.scatter(x[columns[j-1]],y)

#Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[1])]
pd.DataFrame({'vif':vif[0:]}, index = x_train.columns).T

#Normality
sns.distplot(residual)

#PP
import scipy as sp
fig, ax = plt.subplots(figsize=(6,2.5))
_, (__, ___, r) = sp.stats.probplot(residual, plot=ax, fit=True)

#QQ plot
import statsmodels.api as sm
import pylab as py
sm.qqplot(residual, fit=True, line = '45')
py.show()
np.mean(residual)

#homoskedasticity

fig, ax = plt.subplots(figsize=(6,2.5))
ax.scatter(y_pred, residual)
plt.scatter(y_pred, residual)

#autocorrelation

import statsmodels.api as sm
import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(residual, lags=40 , alpha=0.05)
acf.show()

data.corr()['charges'].sort_values()

#correlation matrix

x_cor = x.corr()


plt.subplots(figsize=(10,10))
sns.set(font_scale=1)
sns.heatmap(x_cor, linewidths=3,fmt='.2f', annot=True)
plt.show()

#Changed model based on assumptions
#updated data after linear regression assumptions
x_asmp = x.drop(['bmi'], axis=1)

#reapply linear regression and see if any progress
from sklearn.model_selection import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(x_asmp, y, test_size = 0.25, random_state = 0)

#Linear Regression Model
from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(x_tr, y_tr)

# Predict
y_pr = regress.predict(x_te)

#Accuracy
from sklearn.metrics import r2_score
r2_te = r2_score(y_te, y_pr)

from sklearn.metrics import mean_squared_error
from math import sqrt
mse_te = sqrt(mean_squared_error(y_te, y_pr))
print(r2_te)
print(mse_te)

#apply other models
#polynomial regression

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(x_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

# Predicting a new result with Polynomial Regression
poly_pred = lin_reg_2.predict(poly_reg.fit_transform(x_test))

r2_poly = r2_score(y_test, poly_pred)

print(r2_poly)

#Decision tree
# decision tree
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test)

r2_dt = r2_score(y_test, y_pred)
mse_dt = sqrt(mean_squared_error(y_test, y_pred))

print(r2_dt)
print(mse_dt)

#random forest
# random forest
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10000, random_state=0, max_depth=5)
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test)

r2_rf = r2_score(y_test, y_pred)
mse_rf = sqrt(mean_squared_error(y_test, y_pred))

print(r2_rf)
print(mse_rf)

# SVR
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test)

r2_svr = r2_score(y_test, y_pred)
mse_svr = sqrt(mean_squared_error(y_test, y_pred))

print(r2_svr)
print(mse_svr)

#Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_norm = scaler.fit_transform(x)
x_norm = pd.DataFrame(x_norm)
x_norm.columns = x.columns
x_norm.head()

#Apply linear regression
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(x_norm, y, test_size = 0.25, random_state = 0)

#Linear Regression Model
from sklearn.linear_model import LinearRegression
regres = LinearRegression()
regres.fit(xtr, ytr)

# Predict
ypr = regres.predict(xte)

#Accuracy
from sklearn.metrics import r2_score
r2te = r2_score(yte, ypr)

from sklearn.metrics import mean_squared_error
from math import sqrt
msete = sqrt(mean_squared_error(yte, ypr))
print(r2te)
print(msete)

# support vector on scaled data
regressor = SVR(kernel='rbf')
regressor.fit(xtr, ytr)

# Predicting a new result
y_pred = regressor.predict(xte)

r2_svr = r2_score(yte, ypr)
mse_svr = sqrt(mean_squared_error(yte, ypr))

print(r2_svr)
print(mse_svr)


