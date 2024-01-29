"""
#IMPORTAMOS LAS LIBRERIAS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode, mean

"""#CARGAMOS LOS DATOS
* La variable de salida sera Genero y las de entrada seran todas las demas
"""

#base de datos sobre naciminetos en el valle del cauca
data = pd.read_csv("nacimientos.csv",sep=",")
data = data.drop(["Municipio Nacimiento","Parto Atendido por...","País de Residencia","Tipo de Documento de la Madre","Pertenencia Étnica","Régimen Seguridad","Departamento Residencia","Tipo de Administración","Área de Residencia","Municipio Residencia","Nivel Educativo de la Madre","Nivel Educativo de la Madre1","Nombre de la Administradora","Edad del Padre","Número de Hijos Nacidos Vivos","Fecha Anterior del Hijo Nacido Vivo","Nivel Educativo del Padre","APGAR1","APGAR2","Número de Consultas Prenatales","Último Año Aprobado del Padre","Departamento Nacimiento","Área Nacimiento","Estado Conyugal Madre","Rural Disperso","Centro Poblado","Fecha de Nacimiento","Hora de Nacimiento","Barrio","Dirección"],axis=1)
#data.info()

#data.head()

"""#TRANSFORMACIÓN DE LOS DATOS"""

categorias = ["Género","Tipo de Parto","Multiplicidad de Embarazo","Grupo Sanguíneo","Factor RH"]

data["Grupo Sanguíneo"].replace({'NEGATIVO': None, 'POSITIVO': None}, inplace=True)
data["Factor RH"].replace({'NINGUNO DE LOS ANTERIORES': None}, inplace=True)
data["Talla"].replace({'04/08/2016 12:00:00 AM': None}, inplace=True)
data["Multiplicidad de Embarazo"].replace({'1': None,'4': None,'6': None}, inplace=True)
data["Número de Embarazos"].replace({'CONTRIBUTIVO': None,'NO ASEGURADO': None,'SUBSIDIADO': None}, inplace=True)
data["Edad de la Madre"].replace({'ESTÁ SOLTERA': None,'NO ESTÁ CASADA Y LLEVA DOS AÑOS O MÁS VIVIENDO CON SU PAREJA': None, 'NO ESTÁ CASADA Y LLEVA MENOS DE DOS AÑOS VIVIENDO CON SU PAREJA': None}, inplace=True)

data.Peso[data["Peso"]>10] = None

data["Peso"] = pd.to_numeric(data["Peso"],errors='coerce')
data["Número de Embarazos"] = pd.to_numeric(data["Número de Embarazos"],errors='coerce')
data["Edad de la Madre"] = pd.to_numeric(data["Edad de la Madre"],errors='coerce')
data["Talla"] = pd.to_numeric(data["Talla"],errors='coerce')

Mul_Embarazo = str(mode(data["Multiplicidad de Embarazo"]))
Grup_San = str(mode(data["Grupo Sanguíneo"]))
Factor_RH = str(mode(data["Factor RH"]))

from sklearn.impute import SimpleImputer
numericalImputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
categoricalImputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

data['Peso'] = numericalImputer.fit_transform(data[['Peso']])
data["Edad de la Madre"] = numericalImputer.fit_transform(data[['Edad de la Madre']])
data["Número de Embarazos"] = numericalImputer.fit_transform(data[['Número de Embarazos']])
data['Talla'] = numericalImputer.fit_transform(data[['Talla']])

data["Multiplicidad de Embarazo"].replace({None: Mul_Embarazo}, inplace=True)
data["Grupo Sanguíneo"].replace({None: Grup_San}, inplace=True)
data["Factor RH"].replace({None: Factor_RH}, inplace=True)

for column in categorias:
  data[column] = data[column].astype("category")

data.info()

#data.isnull().sum()

data.to_excel("nuevo nacimiento.xlsx")

"""#VISUALIZACIÓN DE LOS DATOS"""

for column in categorias:
  data[column].value_counts().plot(kind = 'barh')
  plt.title(column)
  plt.show()

for column in data.columns:
  if column not in categorias:
    data[column].plot(kind="box")
  plt.show()

"""#CREAMOS LAS DOMMIES"""

data = pd.get_dummies(data,columns=['Tipo de Parto','Grupo Sanguíneo'],drop_first=False)
data = pd.get_dummies(data,columns=['Factor RH','Multiplicidad de Embarazo'],drop_first=True)
data.head()

"""#CODIFICAMOS LA VARIABLE OBJETIVO"""

from sklearn.preprocessing import  LabelEncoder
labelencoder = LabelEncoder()
data["Género"] = labelencoder.fit_transform(data["Género"])
data["Género"]

data.info()

"""#DIVISION 70-30"""

from sklearn.model_selection import train_test_split

X = data.drop("Género", axis = 1) # variables predictoras
Y = data["Género"] #variable objetivo
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.3, stratify = Y)
Ytrain.value_counts().plot(kind = "bar")

#train, test = train_test_split(data, test_size = 0.40,random_state=98)
#test, crossVal = train_test_split(test, test_size=0.50,random_state=98)

from sklearn.preprocessing import StandardScaler
#data1=data
scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain=scaler.transform(Xtrain)
#X_train=pd.DataFrame(data,columns=data1.columns.values)

"""#ENTRENANDO EL MODELO"""

from sklearn.neural_network import MLPClassifier

clf1 = MLPClassifier(solver='lbfgs', learning_rate_init=0.001, alpha=1e-5,
                     hidden_layer_sizes=(), random_state=123, activation="logistic",max_iter=2000) #alpha, regularización, solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
clf1.fit(Xtrain,Ytrain)

#exactitud, precisión, sensibilidad, f1 score
#evaluacion
from sklearn import metrics

clf1_Ypred = clf1.predict(Xtest)
clf1_acc = metrics.accuracy_score(Ytest, clf1_Ypred)
clf1_f1 = metrics.f1_score(Ytest,clf1_Ypred)
clf1_pre = metrics.precision_score(Ytest, clf1_Ypred)
clf1_rec = metrics.recall_score(Ytest,clf1_Ypred)

print("Exactitud: ",clf1_acc," Presición: ",clf1_pre," f1 Score: ",clf1_f1," Sensibilidad: ",clf1_rec)

"""* Creo que el accuracy es la mejor opcion mas directa, porque manejo pocos datos y estan balanceados."""

import seaborn as sns
from sklearn.metrics import confusion_matrix
plt.figure(figsize = (10,10))

plt.title("Matriz de Confusión", fontsize=16)

sns.heatmap(confusion_matrix(Ytest,clf1_Ypred), annot = True, cmap ="crest", fmt = '.0f')

plt.show()
