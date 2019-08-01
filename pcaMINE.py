# PCA

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Wine.csv')
x= dataset.iloc[:,0:13].values
y= dataset.iloc[:,13].values 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x ,y, test_size=0.2 ,random_state=0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)

# Applying PCA
from sklearn.decomposition import PCA
pca= PCA(n_components= 2)
x_train= pca.fit_transform(x_train)
x_test= pca.transform(x_test)
explained_variance= pca.explained_variance_ratio_

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression()
classifier.fit(x_train, y_train)

y_pred= classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
x_set, y_set= x_train, y_train
x1, x2=np.meshgrid(np.arange( start=x_set[:,0].min() -1, stop=x_set[:,0].max() +1,step=0.01),
                   np.arange( start=x_set[:,1].min() -1, stop= x_set[:,1].max() +1, step=0.01))
plt.contourf(x1 ,x2 , classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap= ListedColormap(('red','green','blue')))

plt.xlim(x1.min(), x1.max())
plt.xlim(x2.min(), x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
                c= ListedColormap(('red','green','blue'))(i),label= j)
plt.title('Logistic Regression (Training set)')   
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show() 

# Visualising the Test set results
from matplotlib.colors import ListedColormap
x_set, y_set= x_test, y_test
x1, x2=np.meshgrid(np.arange( start=x_set[:,0].min() -1, stop=x_set[:,0].max() +1,step=0.01),
                   np.arange( start=x_set[:,1].min() -1, stop= x_set[:,1].max() +1, step=0.01))
plt.contourf(x1 ,x2 , classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap= ListedColormap(('red','green','blue')))

plt.xlim(x1.min(), x1.max())
plt.xlim(x2.min(), x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
                c= ListedColormap(('red','green','blue'))(i),label= j)
plt.title('Logistic Regression (Test set)')   
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show() 



































 