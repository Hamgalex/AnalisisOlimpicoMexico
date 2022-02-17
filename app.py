from cmath import nan
from tokenize import Number
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Leer csv's
atletas = pd.read_csv("datasets/athlete_events.csv",encoding='utf-8')
regiones = pd.read_csv("datasets/noc_regions.csv",encoding='utf-8')

# Mergear los dos dataframe
datos = pd.merge(atletas, regiones, on='NOC', how='left')

# Obtener datos México
datos_mexico=datos[(datos.NOC == 'MEX')]

st.title('Análisis de México en las Olimpiadas')
st.header('¿El rendimiento mexicano ha decaído?')
st.text('Resolveremos esta pregunta analizando los datos de los atletas olímpicos')

# Listar a los atletas por verano
st.header('Lista de atletas mexicanos por temporada')

datos_mexico_temporada=datos_mexico[datos_mexico.Season=="Summer"]    
st.dataframe(datos_mexico_temporada)

#
#
# Graficar las medallas por cada deporte
st.header('Medallas por deporte')

medalla_oro=datos_mexico_temporada[datos_mexico_temporada.Medal=='Gold']
medalla_plata=datos_mexico_temporada[datos_mexico_temporada.Medal=='Silver']
medalla_bronce=datos_mexico_temporada[datos_mexico_temporada.Medal=='Bronze']



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
# Graficar las medallas por año
st.header('Medallas por año')
medallas=['Gold','Silver','Bronze']
año_con_medallas=datos_mexico_temporada[datos_mexico_temporada.Medal.isin(medallas)]
años=año_con_medallas.Year.value_counts().reset_index(name='count')
años.rename(columns = {'index':'year'}, inplace = True)
años.sort_values(by="year", inplace=True)

fig = plt.figure(figsize = (10, 5))
plt.scatter(años['year'],años['count'])
plt.title("Años y medallas olímpicas") 
plt.xlabel('Año', fontweight ='bold', fontsize = 15)
plt.ylabel('Número de medallas', fontweight ='bold', fontsize = 15)
st.pyplot(fig)

#
#
# Predecir las medallas en los siguientes años
st.header('Predicción')

regressor = LinearRegression()

X = años.iloc[:,:-1].values  
y = años.iloc[:,1].values 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)
regressor.fit(X_train, y_train)

fig = plt.figure(figsize = (10, 5))
p1=plt.scatter(X, y) 
p2=plt.plot(X_train, regressor.predict(X_train), color='firebrick') 



plt.title("Años y medallas olímpicas") 
plt.xlabel('Año', fontweight ='bold', fontsize = 15)
plt.ylabel('Número de medallas', fontweight ='bold', fontsize = 15)

st.pyplot(fig)


#
#
# Regresión polinómica

lin_reg=LinearRegression()
lin_reg.fit(X,y)

poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)


X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title("Años y medallas olímpicas") 
plt.xlabel('Año', fontweight ='bold', fontsize = 15)
plt.ylabel('Número de medallas', fontweight ='bold', fontsize = 15)
plt.show()

st.pyplot(fig)
