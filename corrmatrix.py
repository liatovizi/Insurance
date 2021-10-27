import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('insurance.csv')
print(df.head())
print(df.describe())


y = df.charges
x= df.drop(['charges'], axis=1)
x.smoker = x.smoker.eq('yes').mul(1)
x.sex = x.sex.eq('female').mul(1)
cv = list(x.select_dtypes(include=["object"]))
print(cv)
x = pd.get_dummies(x, columns=cv, drop_first = True)
print(x.head())
print(df.corr()['charges'].sort_values())

x_cor = x.corr()

plt.subplots(figsize=(10,10))
sns.set(font_scale=1)
sns.heatmap(x_cor, linewidths=3,fmt='.2f', annot=True)
plt.show()
























