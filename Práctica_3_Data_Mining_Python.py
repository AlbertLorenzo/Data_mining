#!/usr/bin/env python
# coding: utf-8

# # Librerías

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2


# # Dataset

# In[2]:


raw_data = pd.read_csv('heart.csv', sep = ',')
data = raw_data.copy()


# # Información del dataset

# In[3]:


data.info()
data


# # Normalización de características no cuantitativas

# In[4]:


qualitative_cols = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']

encoder = LabelEncoder()
for i in qualitative_cols:
    data[i] = encoder.fit_transform(data[i])


# # Transformación de valores negativos

# In[5]:


def rename(num):
    if num < 0:
        return 0
    return num

data['oldpeak'] = data['Oldpeak'].map(rename)
data.drop('Oldpeak', axis = 1, inplace = True)

column_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'oldpeak', 'ST_Slope', 'HeartDisease']

data = data.reindex(columns = column_names)


# # Corrección de valores

# In[6]:


median = data['Cholesterol'].median()
data['Cholesterol'].replace({0 : median}, inplace = True)


# # Separación de valores y variables

# In[7]:


x = data.drop(['HeartDisease'], axis = 1)
y = data['HeartDisease']


# In[8]:


data


# # Gráfico correlación

# In[9]:


px.imshow(data.corr(), title = "Gráfico de correlación")


# # Selección de características

# In[10]:


select = SelectKBest(f_classif, k = 'all').fit(x, y)

Selected_feature_names = x.columns[select.get_support()]

Selected_feature_names


# In[11]:


select.scores_


# # Clustering

# In[12]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[13]:


k_data = data[['Age', 'Cholesterol']]
SSE = []

for cluster in range(1, 11):
    kmeans = KMeans(n_clusters = cluster)
    kmeans.fit(k_data)
    SSE.append(kmeans.inertia_)


# In[14]:


plt.plot(range(1, 11), SSE)
plt.title('Número óptimo de clústers')
plt.xlabel('N clusters')
plt.ylabel('SSE')
plt.show()


# In[15]:


kmeans = KMeans(n_clusters = 2)
kmeans.fit(k_data)


# In[16]:


centroids = kmeans.cluster_centers_
labels = kmeans.labels_

colors = ['r.', 'g.']
for i in range(len(k_data)):
    plt.plot(k_data.iloc[i,0], k_data.iloc[i,1], colors[labels[i]], markersize = 10)
    
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()


# # Regresión logística

# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[18]:


log_reg = LogisticRegression(max_iter = 1000)
log_reg.fit(x.values, y)


# In[19]:


log_reg.score(x.values, y)


# In[20]:


log_reg.predict(np.array([[59, 1, 2, 180, 213, 0, 1, 100, 0, 0.0, 2]]))[0]


# In[21]:


predictions = log_reg.predict(x.values)
cm = confusion_matrix(y, predictions, labels = log_reg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = log_reg.classes_)
disp.plot()


# # Árbol de decisión

# In[22]:


from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[23]:


model = DecisionTreeClassifier(max_depth = 6)
model.fit(x, y)


# In[24]:


plt.figure(figsize = (250, 22))
plot_tree(decision_tree = model, feature_names = x.columns, filled = True, fontsize = 32)


# In[25]:


y_pred = model.predict(x)

print('Positivos reales:', (data['HeartDisease'] == 1).sum(), '\nPositivos del modelo:', (y_pred == 1).sum())
print('Negativos reales:', (data['HeartDisease'] == 0).sum(), '\nNegativos del modelo:', (y_pred == 0).sum())


# In[26]:


print('Precisión del modelo: ', (data['HeartDisease'] == y_pred).sum()/918)


# # Asociación

# In[27]:


from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelBinarizer


# In[28]:


data


# In[29]:


def age(row):
    if row['Age'] < 30:
        return 'Young'
    if row['Age'] >= 30 and row['Age'] <= 60:
        return 'Adult'
    if row['Age'] > 60:
        return 'Old'
    return 0

def resting_bp(row):
    if row['RestingBP'] >= 140:
        return 'High'
    if row['RestingBP'] < 140:
        return 'Low'
    
def cholesterol(row):
    if row['Cholesterol'] >= 200 and row['Cholesterol'] <= 239:
        return 'High'
    elif row['Cholesterol'] > 239:
        return 'Normal-High'
    else:
        return 'Normal'

def heart_rate(row):
    if row['MaxHR'] <= 100:
        return 'Normal'
    if row['MaxHR'] > 100:
        return 'High'


# In[30]:


data['age'] = data.apply(lambda row: age(row), axis = 1)
data['restingBP'] = data.apply(lambda row: resting_bp(row), axis = 1)
data['cholesterol'] = data.apply(lambda row: cholesterol(row), axis = 1)
data['maxHR'] = data.apply(lambda row: heart_rate(row), axis = 1)


# In[31]:


data.drop(['Age', 'MaxHR', 'RestingBP', 'Cholesterol', 'oldpeak'], axis = 1, inplace = True)
data = pd.get_dummies(data, columns = ['age', 'restingBP', 'cholesterol', 'maxHR', 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'FastingBS', 'HeartDisease'])


# In[32]:


frequent_set = apriori(data, min_support = 0.6, use_colnames = True)
frequent_set


# In[33]:


rules = association_rules(frequent_set, metric = 'confidence', min_threshold = 0.6)
rules


# In[ ]:




