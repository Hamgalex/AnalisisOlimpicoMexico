from cmath import nan
from tokenize import Number
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

# Leer csv's
atletas = pd.read_csv("datasets/athlete_events.csv",encoding='utf-8')
regiones = pd.read_csv("datasets/noc_regions.csv",encoding='utf-8')

# Mergear los dos dataframe
datos = pd.merge(atletas, regiones, on='NOC', how='left')

# Obtener datos México
datos_mexico=datos[(datos.NOC == 'MEX')]

st.title('Análisis de México en las Olimpiadas')
st.header('¿El rendimiento mexicano ha decaído?')
st.write('Resolveremos esta pregunta analizando los datos de los atletas olímpicos, primero veremos los datos de todos los atletas olímpicos mexicanos a través de la historia.')

# Listar a los atletas por verano
st.header('Lista de atletas mexicanos por temporada')

datos_mexico_temporada=datos_mexico[datos_mexico.Season=="Summer"]    
st.dataframe(datos_mexico_temporada)

nombremedallas=['Gold','Silver','Bronze']

medallistas=datos_mexico_temporada[datos_mexico_temporada.Medal.isin(nombremedallas)]

medalla_oro_original=datos_mexico_temporada[datos_mexico_temporada.Medal=='Gold']
medalla_plata_original=datos_mexico_temporada[datos_mexico_temporada.Medal=='Silver']
medalla_bronce_original=datos_mexico_temporada[datos_mexico_temporada.Medal=='Bronze']

medalla_oro = medalla_oro_original.drop_duplicates(
  subset = ['Year', 'Event'],
  keep = 'last').reset_index(drop = True)

medalla_plata = medalla_plata_original.drop_duplicates(
  subset = ['Year', 'Event'],
  keep = 'last').reset_index(drop = True)

medalla_bronce = medalla_bronce_original.drop_duplicates(
  subset = ['Year', 'Event'],
  keep = 'last').reset_index(drop = True)

st.write("Ahora veremos las medallas por cada año. Hay que tener en cuenta que en los deportes por equipo solamente se cuenta como una medalla.")

#
#
# Graficar las medallas por año
st.header('Medallas por año')
año_medalla_oro=medalla_oro.Year.value_counts().reset_index(name='Gold')
año_medalla_oro.rename(columns = {'index':'año'}, inplace = True)
año_medalla_plata=medalla_plata.Year.value_counts().reset_index(name='Silver')
año_medalla_plata.rename(columns = {'index':'año'}, inplace = True)
año_medalla_bronce=medalla_bronce.Year.value_counts().reset_index(name='Bronze')
año_medalla_bronce.rename(columns = {'index':'año'}, inplace = True)
años= pd.merge(año_medalla_oro, año_medalla_plata, on ='año', how ="outer")
años= pd.merge(años, año_medalla_bronce, on ='año', how ="outer")

años.fillna(value = 0,
          inplace = True)
años.sort_values(by="año", inplace=True)

año_y_medalla=años['año']
num_bronce_por_año=años['Bronze'].to_list()
num_plata_por_año=años['Silver'].to_list()
num_oro_por_año=años['Gold'].to_list()
lista_para_años = np.array([num_bronce_por_año,num_plata_por_año])

lista_para_años=np.sum(lista_para_años, axis=0)

fig = plt.figure(figsize = (10, 5))
b1 = plt.bar(año_y_medalla, num_bronce_por_año,3,color='goldenrod')
b2 = plt.bar(año_y_medalla, num_plata_por_año,3,bottom=num_bronce_por_año,color='silver')
b3 = plt.bar(año_y_medalla, num_oro_por_año,3,bottom=lista_para_años,color='gold')
plt.legend([b1, b2,b3], ["Bronce", "Plata","Oro"], title="Medallas", loc="upper left")
plt.title("Medallas olímpicas por año") 
plt.xlabel('Número de medallas', fontweight ='bold', fontsize = 15)
plt.ylabel('Año', fontweight ='bold', fontsize = 15)
st.pyplot(fig)



#
#
# Graficar las medallas por cada deporte
st.header('Medallas por deporte')

deportes_medalla_oro=medalla_oro.Sport.value_counts().reset_index(name='Gold')
deportes_medalla_oro.rename(columns = {'index':'deporte'}, inplace = True)
deportes_medalla_plata=medalla_plata.Sport.value_counts().reset_index(name='Silver')
deportes_medalla_plata.rename(columns = {'index':'deporte'}, inplace = True)
deportes_medalla_bronce=medalla_bronce.Sport.value_counts().reset_index(name='Bronze')
deportes_medalla_bronce.rename(columns = {'index':'deporte'}, inplace = True)

medallas= pd.merge(deportes_medalla_oro, deportes_medalla_plata, on ='deporte', how ="outer")
medallas= pd.merge(medallas, deportes_medalla_bronce, on ='deporte', how ="outer")

medallas.fillna(value = 0,
          inplace = True)

deportes=medallas['deporte']
numbronce=medallas['Bronze'].to_list()
numplata=medallas['Silver'].to_list()
numoro=medallas['Gold'].to_list()

my_list = np.array([numbronce,numplata])
my_list=np.sum(my_list, axis=0)

fig = plt.figure(figsize = (6, 7))
b1 = plt.barh(deportes, numbronce,color='goldenrod')
b2 = plt.barh(deportes, numplata,left=numbronce,color='silver')
b3 = plt.barh(deportes, numoro,left=my_list,color='gold')
plt.legend([b1, b2,b3], ["Bronce", "Plata","Oro"], title="Medallas", loc="upper right")
plt.title("Medallas olímpicas por deporte") 
plt.xlabel('Número de medallas', fontweight ='bold', fontsize = 15)
plt.ylabel('Deporte', fontweight ='bold', fontsize = 15)

st.pyplot(fig)

#
#
# Distribución de edades por cada medalla

# Oro
medallistas_edades_distribucion = medallistas[np.isfinite(medallistas['Age'])]
fig = plt.figure(figsize=(10, 5))
plt.tight_layout()
sns.countplot(medallistas_edades_distribucion['Age'])
plt.title('Distribución de Edad de Medallistas')
st.pyplot(fig)


#
#
# Sexos a través del tiempo

medallistas_hombres = datos_mexico_temporada[datos_mexico_temporada.Sex == 'M']
medallistas_mujeres = datos_mexico_temporada[datos_mexico_temporada.Sex == 'F']

# Hombres
hombres_grafica = medallistas_hombres.groupby('Year')['Sex'].value_counts()
fig= plt.figure(figsize=(10, 5))
plt.plot(hombres_grafica.loc[:,'M'])
plt.title('Variación de atletas hombres a través del tiempo')
st.pyplot(fig)


# Mujeres
mujeres_grafica = medallistas_mujeres.groupby('Year')['Sex'].value_counts()
fig= plt.figure(figsize=(10, 5))
plt.plot(mujeres_grafica.loc[:,'F'])
plt.title('Variación de atletas mujeres a través del tiempo')
st.pyplot(fig)

#
#
# Variación de Peso a través de los años

# Hombres
fig= plt.figure(figsize=(10, 5))
sns.pointplot('Year', 'Weight', data=medallistas_hombres)
plt.title('Variación del Peso de atletas hombres a través del tiempo')
st.pyplot(fig)


# Hombres
fig= plt.figure(figsize=(10, 5))
sns.pointplot('Year', 'Weight', data=medallistas_mujeres)
plt.title('Variación del Peso de atletas Mujeres a través del tiempo')
st.pyplot(fig)


#
#
# Variación de altura a través de los años

# Hombres
fig= plt.figure(figsize=(10, 5))
sns.pointplot('Year', 'Height', data=medallistas_hombres, palette='Set2')
plt.title('Variation of Height for Male Athletes over time')
st.pyplot(fig)

# Mujeres
fig= plt.figure(figsize=(10, 5))
sns.pointplot('Year', 'Height', data=medallistas_mujeres, palette='Set2')
plt.title('Variation of Height for Female Athletes over time')
st.pyplot(fig)

#
#
# Predecir las medallas en los siguientes años
st.header('Predicción')

regressor = LinearRegression()
X = años['año'].values.reshape(-1, 1)
y1 = años['Gold'].values
y2 = años['Silver'].values
y3 = años['Bronze'].values
X_train, X_test, y_train, y_test = train_test_split(X,y1,random_state=0)
regressor.fit(X_train, y_train)

#y1
fig = plt.figure(figsize = (10, 5))
p1=plt.scatter(X, y1,color="gold") 
p2=plt.plot(X_train, regressor.predict(X_train), color='darkgoldenrod') 
plt.title("Años y medallas de Oro") 
plt.xlabel('Año', fontweight ='bold', fontsize = 15)
plt.ylabel('Número de medallas', fontweight ='bold', fontsize = 15)
plt.ylim([0.5, 4])
st.pyplot(fig)

# y2
fig = plt.figure(figsize = (10, 5))
X_train, X_test, y_train, y_test = train_test_split(X,y2,random_state=0)
regressor.fit(X_train, y_train)
p1=plt.scatter(X, y2,color="silver") 
p2=plt.plot(X_train, regressor.predict(X_train), color='grey') 
plt.title("Años y medallas de Plata") 
plt.xlabel('Año', fontweight ='bold', fontsize = 15)
plt.ylabel('Número de medallas', fontweight ='bold', fontsize = 15)
plt.ylim([0.5, 4])
st.pyplot(fig)

#y3
fig = plt.figure(figsize = (10, 5))
X_train, X_test, y_train, y_test = train_test_split(X,y3,random_state=0)
regressor.fit(X_train, y_train)
p1=plt.scatter(X, y3,color="goldenrod") 
p2=plt.plot(X_train, regressor.predict(X_train), color='saddlebrown') 
plt.title("Años y medallas de Bronce") 
plt.xlabel('Año', fontweight ='bold', fontsize = 15)
plt.ylabel('Número de medallas', fontweight ='bold', fontsize = 15)
plt.ylim([0.5, 4])
st.pyplot(fig)


