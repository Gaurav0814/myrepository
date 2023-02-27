import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

d2 = pd.read_csv(r'C:\Users\gaura\AppData\Local\Temp\Temp2_OneDrive_2022-12-18 (3).zip\Data Science Projects 50 - Real Estate Price Prediction\Real estate.csv')
# head = d2.head(10)
# print(head)
# Basic EDA
# info = d2.info()
# print(info)
tbd = ['X1 transaction date','No']
d2 = d2.drop(tbd, axis=1)
des = d2.describe()
three = d2.head(3)
print(d2)
print(three)
print(des)
image = d2.hist(figsize=(10,10))
corr = d2.corr()
print(corr)
fig, ax = plt.subplots(figsize=(22,15))
sns.heatmap(corr, annot=True, ax=ax)
# X3 distance to the nearest MRT station' column shows least correlation
# emp = d2.isnull().sum()
# print(emp)

# Numerical attributes comparison using scatterplot
sns.barplot(x=d2['X4 number of convenience stores'], y=d2['Y house price of unit area'])
sns.relplot(x=d2['X2 house age'], y=d2['Y house price of unit area'])
sns.relplot(x=d2['X3 distance to the nearest MRT station'], y=d2['Y house price of unit area'])
sns.lineplot(x=d2['X2 house age'], y=d2['X3 distance to the nearest MRT station'])
# This shows that houses with an average age of <b>15 - 20 years</b> have <b>high distances</b> to MRT station while the houses aged for <b>35+</b> years are <b>more closer</b> to the stations"
### Conclusion from EDA and Graph plots:\n",
# 1. Data is clean having no null values<br>
# 2. Data doesn't have High correlation amongst attributes<br>
# 3. Houses with more convenience stores in the area, with low age have high prices<br>
# 4. Houses that are aged have more MRT stations near them and fall in low price.

# Outlier Detection
plt.figure(figsize=(13,5))
for feat, grd in zip(d2, range(231,237)):
    plt.subplot(grd)
    sns.boxplot(y=d2[feat], color='grey')
    plt.ylabel('Value')
    plt.title('Boxplot\\n%s'%feat)
plt.tight_layout()

from sklearn.model_selection import train_test_split

X2 = d2.loc[:,'X2 house age' : 'X6 longitude']
y2 = d2.loc[:,'Y house price of unit area']
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X2, y2, test_size=0.2, random_state=1)
print(X_train_2.shape, X_test_2.shape)
print(y_train_2.shape, y_test_2.shape)

from statsmodels.graphics.gofplots import qqplot
qqplot(X2,line='s')
# plt.show()

from scipy.stats import skew
print(skew(X2))
# Scaling Data using Min-Max Scaler
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm_2 = mms.fit_transform(X_train_2)
X_test_norm_2 = mms.transform(X_test_2)
# Scaling Data using Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_2)
X_train_std_2 = scaler.transform(X_train_2)
X_test_std_2 = scaler.transform(X_test_2)
# Comparing Scaled Data
xx = np.arange(len(X_train_std_2))
yy1 = X_train_norm_2[:,0]
yy2 = X_train_std_2[:,0]
plt.scatter(xx,yy1,color='b')
plt.scatter(xx,yy2,color='r')
print(X_train_std_2.shape)
print(X_test_std_2.shape)
# plt.show()
# OLS regression analysis
import statsmodels.api as sm
model_ols = sm.OLS(y_train_2, X_train_norm_2)
fitted = model_ols.fit()
print(fitted.summary())

from scipy.stats import shapiro
fig, ax = plt.subplots(figsize=(16,4), ncols=2)
ax[0] = sns.scatterplot(x=y_train_2, y=fitted.resid, ax=ax[0])
ax[1] = sns.histplot(fitted.resid, ax=ax[1])
statistic, p_value = shapiro(fitted.resid)
if p_value>0.05:
    print("Distribution is normal. Statistic: {0:.3}, p-value: {1:.4}".format(statistic, p_value))
else:
    print("Distribution is not normal. Statistic: {0:.3}, p-value: {1:.4}".format(statistic, p_value))

from sklearn.neighbors import KNeighborsRegressor as knn
model4 = knn(n_neighbors=3,p=1,algorithm='brute')
model4.fit(X_train_norm_2,y_train_2)
ypred3 = model4.predict(X_test_norm_2)
print(ypred3)

k_values = np.arange(1,100,2)
train_score_arr = []
val_score_arr = []
for k in k_values:
    model2 = knn(n_neighbors=k,p=1)
    model2.fit(X_train_norm_2,y_train_2)
    train_score = model2.score(X_train_norm_2, y_train_2)
    train_score_arr.append(train_score*100)
    val_score = model2.score(X_test_norm_2, y_test_2)
    val_score_arr.append(val_score*100)
    print("k=%d, train_accuracy=%.2f%%, test_accuracy=%.2f%%" % (k, train_score * 100, val_score*100))

plt.plot(k_values,train_score_arr,'g')
plt.plot(k_values,val_score_arr,'r')
plt.show()

from sklearn.model_selection import cross_val_score
cross_val_score_train = cross_val_score(model4, X_train_norm_2, y_train_2, cv=10, scoring='r2')
print(cross_val_score_train)
avg = cross_val_score_train.mean()
print(avg)

from sklearn.metrics import r2_score
print(r2_score(y_test_2, ypred3))
c = pd.DataFrame(ypred3, columns=['Estimated Price'])
# c = c.head()
print(c)

d = pd.DataFrame(y_test_2)
d = y_test_2.reset_index(drop=True)
# d = d.head()
print(d)

ynew = pd.concat([c,d], axis=1)
print(ynew)
