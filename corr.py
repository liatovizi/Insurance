import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('insurance.csv')
print(df.head())
data1 = df.age
data2 = df.charges

print(data1, data2)

print('data1: mean=%.3f stdv=%.3f' % (np.mean(data1), np.std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (np.mean(data2), np.std(data2)))

plt.scatter(data1, data2)
#plt.show()

covariance = np.cov(data1, data2)
print(covariance)

corr = np.corrcoef(data1, data2)
print(corr)

corr2 = np.corrcoef(df.charges, df.bmi)
print(corr2)

df.smoker = df.smoker.eq('yes').mul(1)
df.sex = df.sex.eq('female').mul(1)

cv = list(df.select_dtypes(include=["object"]))
print(cv)
df = pd.get_dummies(df, columns=cv, drop_first = True)
print(df.head())

df.corr()[['charges']].sort_values(by='charges', ascending=False)


plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(df.corr()[['charges']].sort_values(by='charges', ascending=False), vmin=-1, vmax=1, annot=True, cmap= 'Blues')
heatmap.set_title('Features Correlating with Charges', fontdict={'fontsize':18}, pad=16)

plt.show()