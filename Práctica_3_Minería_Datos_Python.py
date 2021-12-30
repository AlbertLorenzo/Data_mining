#!/usr/bin/env python
# coding: utf-8

# # Práctica 3 - Albert Lorenzo Segarra

# # Librerías

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression
from mlxtend.frequent_patterns import apriori, association_rules


# # Dataset

# In[2]:


raw_data = pd.read_csv('heart.csv', sep = ',')
data = raw_data.copy()


# # Análisis del dataset

# In[3]:


raw_data.info()


# In[4]:


raw_data.sample(5)


# In[5]:


raw_data.isnull().sum()


# In[6]:


raw_data.nunique()


# In[7]:


plt.figure(figsize = (15, 10))
sns.pairplot(raw_data, hue = 'HeartDisease')
plt.title('Distribución de características por pares')
plt.legend('Enfermedad cardiovascular')
plt.tight_layout()
plt.plot()


# # Preparación
# 
# Después de haber leído sobre cómo funcionan más en profunidad algunas ténicas, he decidido que crearé diferentes dataframes para según qué algoritmos.
# 
# En caso de querer realizar una selección de características, utilizaré el dataset original pero con ciertas correcciones.
# 
# Para los algoritmos que no se basan en árboles, categorizaré dichas características mientras que para los que sí se basan en árboles, sencillamente utilizaré el labelEncoder para convertir valores 0..n y transformarlos.
# 
# ## Para técnicas no basadas en árboles

# In[8]:


string_col = data.select_dtypes(include = 'object').columns

data[string_col] = data[string_col].astype('string')

df_nontree = pd.get_dummies(data, columns = string_col, drop_first = False)

df_nontree.drop(['HeartDisease'], axis = 1, inplace = True)

df_nontree = pd.concat([df_nontree , data['HeartDisease']], axis = 1)

df_nontree.head()


# In[9]:


x_nontree = df_nontree.drop(['HeartDisease'], axis = 1)

y_nontree = df_nontree['HeartDisease']


# ## Para técnicas en árboles

# In[10]:


df_tree = data.apply(LabelEncoder().fit_transform)

df_tree.head()


# In[11]:


x_tree = df_tree.drop(['HeartDisease'], axis = 1)

y_tree = df_tree['HeartDisease']


# # Selección de características
# 
# Después de haber preparado previamente los diferentes datasets, utilizaré uno nuevo para la selección de características y la correlación de Pearson.
# 
# Lo primero será transformar los datos nominales, después corregiré los valores negativos y finalmente, trataré el colesterol ya que por algún motivo algunos de los pacientes clínicos tienen un colesterol 0 y una persona no puede tener ese valor en dicha característica, así que lo reemplazaré por la mediana en lugar de la media.

# ## Transformación datos categóricos

# In[12]:


qualitative_cols = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']

encoder = LabelEncoder()
for i in qualitative_cols:
    raw_data[i] = encoder.fit_transform(data[i])


# ## Corrección de valores negativos

# In[13]:


def rename(num):
    if num < 0:
        return 0
    return num

raw_data['oldpeak'] = raw_data['Oldpeak'].map(rename)
raw_data.drop('Oldpeak', axis = 1, inplace = True)

column_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'oldpeak', 'ST_Slope', 'HeartDisease']

raw_data = raw_data.reindex(columns = column_names)


# ## Correción de características

# In[14]:


median = raw_data['Cholesterol'].median()
raw_data['Cholesterol'].replace({0 : median}, inplace = True)


# # Gráfico correlación
# 
# Este gráfico nos indica, omitiendo características nominales/ordinales, cuáles son las variables que más influyen sobre heartdisease que es nuestro target.
# 
# Hay dos tipos de relaciones, positiva y negativa. Teniendo un par de valores (a:b), una relación positiva afecta a b de forma proporcional, es decir, si el valor de a sube, el de b también. Para las relaciones negativas es lo contrario, si a baja, b sube.

# In[15]:


px.imshow(raw_data.corr(), title = "Correlación de Pearson")


# # SelectKBest
# 
# En esta parte del código, haré dos selecciones con dos algoritmos diferentes. f_regression para las regresiones y chi2 para las classificaciones.

# In[16]:


x = raw_data.drop(['HeartDisease'], axis = 1)

y = raw_data['HeartDisease']


# In[17]:


skb_reg = SelectKBest(f_regression, k = 'all').fit(x, y)
skb_class = SelectKBest(chi2, k = 'all').fit(x, y)

feat_names = x.columns[skb_reg.get_support()]  


# In[18]:


df_features_data = {'Características': feat_names, 'P. Regresión': skb_reg.scores_, 'P. clasificación': skb_class.scores_}

df_features = pd.DataFrame(df_features_data)

df_features


# # Clustering KMeans
# 
# Para KMeans podemos realizar el método del codo, el cuál nos indica el número óptimo de clusters que necesitamos. 

# In[19]:


k_data = raw_data[['MaxHR', 'Age']]
k = range(1,12)
SSE = []

for cluster in k:
    kmeans = KMeans(n_clusters = cluster)
    kmeans.fit(k_data)
    SSE.append(kmeans.inertia_)


# In[20]:


cl = pd.DataFrame({'Clusters': k, 'SSE': SSE})
fig = (px.line(cl, x = 'Clusters', y = 'SSE', template = 'seaborn')).update_traces(mode = 'lines+markers')
fig.show()


# In[21]:


kmeans = KMeans(n_clusters = 3)
kmeans.fit(k_data)


# In[22]:


data['cluster_group'] = kmeans.labels_

fig = px.scatter(data, x = 'Age', y = 'MaxHR',  color = 'cluster_group', hover_data = ['HeartDisease'], template = 'ggplot2')

fig.show()


# In[23]:


data.drop(['cluster_group'], axis = 1, inplace = True)


# # Regresión logística

# In[24]:


log_reg = LogisticRegression(max_iter = 1000)
log_reg.fit(x_nontree.values, y_nontree)


# In[25]:


log_reg.score(x_nontree.values, y_nontree)


# In[26]:


test_data = x_nontree.sample()
test_data


# In[27]:


# Ejemplo de predicción con el propio sample, aunque los datos se pueden introducir manualmente. 1 = positivo en heartdisease, 0 = negativo

print('Predicción heart disease: ', log_reg.predict(np.array(test_data))[0])


# In[28]:


predictions = log_reg.predict(x_nontree.values)
cm = confusion_matrix(y_nontree, predictions, labels = log_reg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = log_reg.classes_)
disp.plot()


# Con esta matriz podemos ver cuántos casos son realmente positivos y negativos. 
# 
# En mi caso, tengo 342 (verdaderos negativos) + 460 (verdaderos negativos) predicciones correctas y 48 + 68 prediciones incorrectas. Si observamos la precisión del modelo, que es del 87% y sabiendo la cantidad de items con el que cuenta el dataset, 918, sabemos que fallará en un 13% que coincide (casi) con las 48 + 68 predicciones incorrectas, es decir, el 13% de esos 918 se le ha hecho una predicción errónea.

# # Árbol de decisión

# In[29]:


from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[30]:


model = DecisionTreeClassifier(max_depth = 6)
model.fit(x_tree, y_tree)


# In[31]:


plt.figure(figsize = (250, 22))
plot_tree(decision_tree = model, feature_names = x_tree.columns, filled = True, fontsize = 32)


# In[32]:


y_pred = model.predict(x_tree)

print('Positivos reales:', (data['HeartDisease'] == 1).sum(), '\nPositivos del modelo:', (y_pred == 1).sum())
print('Negativos reales:', (data['HeartDisease'] == 0).sum(), '\nNegativos del modelo:', (y_pred == 0).sum())


# In[33]:


print('Precisión del modelo: ', (data['HeartDisease'] == y_pred).sum()/918)


# # Reglas de asociación mediante apriori
# 
# Para obtener reglas de asociación, primero necesitaré preparar el dataset así que discretizaré todas las características. Pero primero tendré que categorizar ciertos valores para posteriormente, crear sus columnas de identidad.
# 
# En mi caso, lo primero será hacer este proceso con los parámetros {age, resting_bp, cholesterol, heart_rate}. Comentar que he escogido unas estimaciones genéricas.

# In[34]:


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
    
def oldpeak_levels(row):
    if row['oldpeak'] < 2:
        return 'Low'
    if row['oldpeak'] >= 2 and row['oldpeak'] <= 2.5:
        return 'Risk'
    if row['oldpeak'] > 2.5:
        return 'Terrible'


# In[35]:


discrete_dict = {'age': age, 'restingBP':resting_bp, 'cholesterol': cholesterol, 'maxHR': heart_rate, 'oldpeak': oldpeak_levels}

for key in discrete_dict:
    raw_data[key] = raw_data.apply(lambda row: discrete_dict[key](row), axis = 1)


# In[36]:


raw_data.drop(['Age', 'MaxHR', 'RestingBP', 'Cholesterol'], axis = 1, inplace = True)
raw_data = pd.get_dummies(raw_data, columns = ['age', 'restingBP', 'cholesterol', 'maxHR', 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'FastingBS', 'oldpeak', 'HeartDisease'])


# In[37]:


discretized_data = raw_data.copy()
discretized_data


# In[38]:


frequent_set = apriori(discretized_data, min_support = 0.6, use_colnames = True)
frequent_set.sort_values(by = ['support'], ascending = False)


# In[39]:


rules = association_rules(frequent_set, metric = 'confidence', min_threshold = 0.7)
rules.head(10).sort_values(by = ['confidence'], ascending = False)


# # Python y Weka
# 
# ## Comparación
# 
# Personalmente, creo que Python es una herramienta con mucho más potencial a la hora de manejar datasets, aplicar técnicas de minería de datos, escoger algoritmos, etc.
# 
# Una de las ventajas que tiene Python es que al ser tan popular, y además ser tan utilizado en el sector de machine learning, se encuentra bastante más información sobre el uso del mismo. También destacar, que las propias librerías están documentadas y explican cada función de forma extensa, los parámetros, la información de entrada y salida o cualquier otro detalle.
# 
# Además, te permite procesar el dataset según consideres lo que es mejor para un algorito u otro. Esto con Weka, según tengo entendido, no es posible.
# 
# Al no haber tratado previamente con este lenguaje he encontrado alguna dificultad a la hora de codificar funcionalidades, pero esto se resuelve fácilmente ya que al haber tantas librerías no tienes que codificar funcionalidades básicas, cosa que en otros lenguajes sí deberías hacer.
# 
# Por otra parte, creo que Weka es muy potente, pero no tiene tanta versatilidad como Python. Además, considero que es bastante más intuitiva al contar con un entorno gráfico y como primer contacto resulta más descriptivo.
# 
# Una de las ventajas que tiene Weka sobre Python es que todo el graficado lo hace de forma automática, mientras que con este último resulta algo más tedioso si no entiendes la herramienta.
# 
# Sobre los modelos y los resultados, creo que los dos son parecidos, al final según los datos y cómo los prepares los algoritmos crearán un modelo u otro.
# 
# ## Conclusiones
# 
# Considero que Python es la herramienta con la que me quedaría ya que al final, te permite tratar con la información de forma manual y sabes en todo momento lo que está ocurriendo.
# 
# Weka la utilizaría en casos más específicos o para aprender.
