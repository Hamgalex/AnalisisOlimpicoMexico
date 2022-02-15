from cmath import nan
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Leer csv's
atletas = pd.read_csv("datasets/athlete_events.csv")
regiones = pd.read_csv("datasets/noc_regions.csv")

# Mergear los dos dataframe
datos = pd.merge(atletas, regiones, on='NOC', how='left')

# Obtener datos México
datos_mexico=datos[(datos.Team == 'Mexico')]

st.title('Análisis de México en las Olimpiadas')
st.header('¿El rendimiento mexicano ha decaído?')
st.text('Resolveremos esta pregunta analizando los datos de los atletas olímpicos')

# Listar a los atletas por invierno o por verano
st.header('Lista de atletas mexicanos por temporada')
option = st.selectbox(
    'Tipo de Olimpiadas',
    ('Verano', 'Invierno'))

if option=='Verano':
    season='Summer'
else:
    season='Winter'

datos_mexico_temporada=datos_mexico[datos_mexico.Season==season]    
st.dataframe(datos_mexico_temporada)

# Graficar las medallas por cada deporte
st.header('Medallas por deporte')

medalla_oro=datos_mexico_temporada[datos_mexico_temporada.Medal=='Gold']
medalla_plata=datos_mexico_temporada[datos_mexico_temporada.Medal=='Silver']
medalla_bronce=datos_mexico_temporada[datos_mexico_temporada.Medal=='Bronze']

deportes_medalla_oro=medalla_oro.Event.value_counts().reset_index(name='Gold')
deportes_medalla_plata=medalla_plata.Event.value_counts().reset_index(name='Silver')
deportes_medalla_bronce=medalla_bronce.Event.value_counts().reset_index(name='Bronze')

medallas=deportes_medalla_oro.set_index('index').join(deportes_medalla_plata.set_index('index'), how='outer').join(deportes_medalla_bronce.set_index('index'), how='outer')

fig = plt.figure(figsize = (10, 4))
plt.xticks(rotation='vertical')
p1 = plt.bar(medallas.index, medallas.Gold,color="gold")
p2= plt.bar(medallas.index, medallas.Silver,color="silver")
p3= plt.bar(medallas.index, medallas.Bronze,color="goldenrod")

plt.legend((p1[0], p2[0],p3[0]), ('Oro', 'Plata','Bronce'))

st.pyplot(fig)