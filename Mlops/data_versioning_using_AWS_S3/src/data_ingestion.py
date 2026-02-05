import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

df = pd.read_csv('D:\Game\ML from Scratch\Mlops\data_versioning_using_AWS_S3\src\student_performance.csv')
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df = df.select_dtypes(include='number')

X = df.drop(columns=['placed'])
y = df['placed']

# scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# applying PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(data=X_pca , columns=['PC1' , 'PC2' , 'PC3'])
df_pca['placed'] =  y.values


df_pca.to_csv(os.path.join('data','processed','student_performance_pca.csv'), index=False)
