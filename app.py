from cmath import nan
from tokenize import Number
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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


medalla_oro=datos_mexico_temporada[datos_mexico_temporada.Medal=='Gold']
medalla_plata=datos_mexico_temporada[datos_mexico_temporada.Medal=='Silver']
medalla_bronce=datos_mexico_temporada[datos_mexico_temporada.Medal=='Bronze']


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

fig = plt.figure(figsize = (10, 5))
plt.scatter(años['año'],años['Gold'],color='gold')
plt.scatter(años['año'],años['Silver'],color='silver')
plt.scatter(años['año'],años['Bronze'],color='goldenrod')
plt.title("Años y medallas olímpicas") 
plt.xlabel('Año', fontweight ='bold', fontsize = 15)
plt.ylabel('Número de medallas', fontweight ='bold', fontsize = 15)
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
p1=plt.scatter(X, y1) 
p2=plt.plot(X_train, regressor.predict(X_train), color='gold') 
plt.title("Años y medallas de Oro") 
plt.xlabel('Año', fontweight ='bold', fontsize = 15)
plt.ylabel('Número de medallas', fontweight ='bold', fontsize = 15)
plt.ylim([0.5, 17])
st.pyplot(fig)

# y2
fig = plt.figure(figsize = (10, 5))
X_train, X_test, y_train, y_test = train_test_split(X,y2,random_state=0)
regressor.fit(X_train, y_train)
p1=plt.scatter(X, y2) 
p2=plt.plot(X_train, regressor.predict(X_train), color='silver') 
plt.title("Años y medallas de Plata") 
plt.xlabel('Año', fontweight ='bold', fontsize = 15)
plt.ylabel('Número de medallas', fontweight ='bold', fontsize = 15)
plt.ylim([0.5, 17])
st.pyplot(fig)

#y3
fig = plt.figure(figsize = (10, 5))
X_train, X_test, y_train, y_test = train_test_split(X,y3,random_state=0)
regressor.fit(X_train, y_train)
p1=plt.scatter(X, y3) 
p2=plt.plot(X_train, regressor.predict(X_train), color='goldenrod') 
plt.title("Años y medallas de Bronce") 
plt.xlabel('Año', fontweight ='bold', fontsize = 15)
plt.ylabel('Número de medallas', fontweight ='bold', fontsize = 15)
plt.ylim([0.5, 17])
st.pyplot(fig)


