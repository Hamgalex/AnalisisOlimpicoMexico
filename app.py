from cmath import nan
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Leer csv's
atletas = pd.read_csv("datasets/athlete_events.csv",encoding='utf-8')
regiones = pd.read_csv("datasets/noc_regions.csv",encoding='utf-8')

# Mergear los dos dataframe
datos = pd.merge(atletas, regiones, on='NOC', how='left')
datos.replace(to_replace="é", value="e")
# Obtener datos México
datos_mexico=datos[(datos.Team == 'Mexico')]

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
deportes_medalla_plata=medalla_plata.Sport.value_counts().reset_index(name='Silver')
deportes_medalla_bronce=medalla_bronce.Sport.value_counts().reset_index(name='Bronze')

medallas=deportes_medalla_oro.set_index('index').join(deportes_medalla_plata.set_index('index'), how='outer').join(deportes_medalla_bronce.set_index('index'), how='outer')

fig = plt.figure(figsize = (6, 8))
p1 = plt.barh(medallas.index, medallas.Gold,color="gold")
p2= plt.barh(medallas.index, medallas.Silver,color="silver")
p3= plt.barh(medallas.index, medallas.Bronze,color="goldenrod")
plt.xlabel('Número de Medallas', fontweight ='bold', fontsize = 15)
plt.ylabel('Deporte', fontweight ='bold', fontsize = 15)
plt.legend((p1[0], p2[0],p3[0]), ('Oro', 'Plata','Bronce'))
st.pyplot(fig)



#
#
# Graficar las medallas por año
st.header('Medallas por año')
medallas=['Gold','Silver','Bronze']
año_con_medallas=datos_mexico_temporada[datos_mexico_temporada.Medal.isin(medallas)]
años=año_con_medallas.Year.value_counts().reset_index(name='Year').set_index('index')
años.sort_values(by="index", inplace=True)
# st.dataframe(años)

fig = plt.figure(figsize = (10, 5))
plt.plot(años.index,años.Year)
st.pyplot(fig)