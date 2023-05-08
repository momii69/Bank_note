import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ds=pd.read_csv("C:\\Users\\Super\\Downloads\\data_banknote_authentication.txt"
               ,names=['variance','skewness','kertosis','Entropy','class'])
x=ds[['variance','skewness','kertosis','Entropy']].values
y=ds['class'].values
xmean=np.mean(x)
xstd=np.std(x)
x=x-xmean
x=x/xstd
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)
Y_pred=model.predict(x_test)
ds['class'].value_counts().plot(kind='bar')
plt.title('Class Distribution for Target Column')
plt.ylabel('count')
plt.xlabel('Class')
plt.show()
ds.hist(bins=25,figsize=(11,9),layout=(2,3))
plt.show()
import seaborn as sns
sns.set_style('darkgrid')
sns.pairplot(ds,hue='class')
plt.show()