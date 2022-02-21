
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Leer csv's
atletas = pd.read_csv("datasets/athlete_events.csv",encoding='utf-8')
regiones = pd.read_csv("datasets/noc_regions.csv",encoding='utf-8')

# Mergear los dos dataframe
datos = pd.merge(atletas, regiones, on='NOC', how='left')

# Obtener datos México
datos_mexico=datos[(datos.NOC == 'MEX')]

st.title('Análisis de México en las Olimpiadas de Verano')
st.header('¿El rendimiento mexicano ha decaído?')
st.write('Resolveremos esta pregunta analizando los datos de los atletas olímpicos, primero veremos los datos de todos los atletas olímpicos mexicanos a través de la historia.')
st.subheader("Paquetes requeridos")

st.code("import streamlit as st\nimport pandas as pd\nimport numpy as np\nfrom matplotlib import pyplot as plt\nimport seaborn as sns\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split")



st.subheader('Lectura de datos')
st.write("Primero leeremos los datos de los csv's, esto es:")
st.code("atletas = pd.read_csv(\"datasets/athlete_events.csv\",encoding='utf-8') \nregiones = pd.read_csv(\"datasets/noc_regions.csv\",encoding='utf-8')")
st.write("El dataframe de atletas tiene todos los atletas que participaron en los juegos olímpicos, mientras que el dataframe de regiones tiene el código de la región con su nombre. Para observar mejor los datos los mergearemos:")

st.code("datos = pd.merge(atletas, regiones, on='NOC', how='left')")

st.subheader("Filtrar los datos")
st.write("Filtraremos los datos para que nos de todos los atletas de méxico, luego filtraremos los restantes dependiendo si sacó medalla de oro, bronce o plata, esto es:")
st.code("datos_mexico=datos[(datos.NOC == 'MEX')]\ndatos_mexico_temporada=datos_mexico[datos_mexico.Season==\"Summer\"]\nnombremedallas=['Gold','Silver','Bronze']\nmedalla_oro_original=datos_mexico_temporada[datos_mexico_temporada.Medal=='Gold']\nmedalla_plata_original=datos_mexico_temporada[datos_mexico_temporada.Medal=='Silver']\nmedalla_bronce_original=datos_mexico_temporada[datos_mexico_temporada.Medal=='Bronze']")
# Listar a los atletas por verano

datos_mexico_temporada=datos_mexico[datos_mexico.Season=="Summer"]    

st.write("Mostraremos los datos de todos los atletas mexicanos: ")
st.dataframe(datos_mexico_temporada)

nombremedallas=['Gold','Silver','Bronze']

medallistas=datos_mexico_temporada[datos_mexico_temporada.Medal.isin(nombremedallas)]

medalla_oro_original=datos_mexico_temporada[datos_mexico_temporada.Medal=='Gold']
medalla_plata_original=datos_mexico_temporada[datos_mexico_temporada.Medal=='Silver']
medalla_bronce_original=datos_mexico_temporada[datos_mexico_temporada.Medal=='Bronze']


st.subheader("Limpiar los datos")
st.write("Existe una situación con los datos anteriores; aparece que en una disciplina por equipo se le da una medalla a cada uno, lo cual es incorrecto. Por ejemplo, en el 2012 México ganó medalla de oro en Fútbol, y en nuestro dataset aparece que cada uno de los 16 integrantes de nuestro equipo tiene una medalla de oro, lo cual infla las estadísticas. Para contrarrestar esto, eliminaremos los registros que tengan los mismos valores de año y disciplina, esto para que solo haya un participante por cada disciplina: ")

with st.echo():
    medalla_oro = medalla_oro_original.drop_duplicates(
      subset = ['Year', 'Event'],
      keep = 'last').reset_index(drop = True)

    medalla_plata = medalla_plata_original.drop_duplicates(
      subset = ['Year', 'Event'],
      keep = 'last').reset_index(drop = True)

    medalla_bronce = medalla_bronce_original.drop_duplicates(
      subset = ['Year', 'Event'],
      keep = 'last').reset_index(drop = True)



st.write("Con esto ya no tendremos medallas infladas. Ahora veremos las medallas por cada año.")

#
#
# Graficar las medallas por año
st.header('Medallas por año')


st.write("Primero necesitamos saber cuantas medallas de oro, plata y bronce hay por cada año, para esto haremos un value counts para las de oro, plata y bronce y despues las mergearemos, esto es:")

with st.echo():

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

st.write(" Ahora necesitamos un arreglo de los años y listas de las medallas de oro, plata y bronce por año, luego graficaremos:")


with st.echo():
  año_y_medalla=años['año']
  num_bronce_por_año=años['Bronze'].to_list()
  num_plata_por_año=años['Silver'].to_list()
  num_oro_por_año=años['Gold'].to_list()

  # Se realiza lo siguiente para poder apilar las medallas de bronce plata y oro en ese orden.
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

st.write("Ahora realizaremos lo mismo pero con el deporte, esto es:")

with st.echo():
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
st.subheader('Predicción')

st.write("Ahora, vamos a predecir como le irá a México en los siguientes años mediante un modelo de machine learning, el modelo será una regresión lineal")
st.write("Primero crearemos una nueva columna en el dataframe de años, en el cual teníamos cuantas medallas de oro, plata y bronce se ganaron por cada año.")
st.write("Para esto, sabemos que no es lo mismo que un atleta gane una medalla de bronce que una de plata, entonces le daremos un punto si es de bronce, tres puntos si es de plata y 5 puntos si es de oro. De esta manera calcularemos la columna total:")
with st.echo():
  años['total'] = (años['Gold']*5 + años['Silver']*3+ años['Bronze'])
  st.dataframe(años)

st.write("Ahora aplicaremos el modelo de regresión lineal, esto es:")

with st.echo():
  regressor = LinearRegression()
  X = años['año'].values.reshape(-1, 1)
  total=años['total'].values

  X_train, X_test, y_train, y_test = train_test_split(X,total,random_state=0)
  regressor.fit(X_train, y_train)
  fig = plt.figure(figsize = (10, 5))
  p1=plt.scatter(X, total) 
  p2=plt.plot(X_train, regressor.predict(X_train), color='firebrick') 
  plt.title("Predicción del rendimiento de México") 
  plt.xlabel('Años', fontweight ='bold', fontsize = 15)
  plt.ylabel('Puntos por año', fontweight ='bold', fontsize = 15)
  st.pyplot(fig)

st.write("Podemos observar que al realizar una regresión lineal nos resultó una recta con una inclinación positiva, esto nos dice que existe una correlación positiva, lo que significa que México ha aumentado su rendimiento a través de los años. Ahora sólamente queda la pregunta: ¿Por qué ocurre esto?")

st.subheader("Respuestas")
st.write("Para poder responder a la pregunta, veremos los cambios de las características principales a través del tiempo, para esto, separaremos por sexo masculino y femenino:")
#
#
# Sexos a través del tiempo

with st.echo():
  atletas_m = datos_mexico_temporada[datos_mexico_temporada.Sex == 'M']
  atletas_f = datos_mexico_temporada[datos_mexico_temporada.Sex == 'F']



#
#
# Variación de Peso a través de los años

st.write("Ahora observaremos las variaciones del peso de los atletas hombres y mujeres:")

with st.echo():
  # Hacer fuente de los ejes mas pequeña
  sns.set(font_scale=0.85)

  # Hombres
  fig= plt.figure(figsize=(10, 5))
  sns.pointplot('Year', 'Weight', data=atletas_m, palette='Set1')
  plt.title('Variación del Peso de atletas Hombres a través del tiempo')
  plt.xlabel('Años', fontweight ='bold', fontsize = 15)
  plt.ylabel('Kilogramos (kg)', fontweight ='bold', fontsize = 15)
  st.pyplot(fig)


  # Mujeres
  fig= plt.figure(figsize=(10, 5))
  sns.pointplot('Year', 'Weight', data=atletas_f, palette='Set1')
  plt.title('Variación del Peso de atletas Mujeres a través del tiempo')
  plt.xlabel('Años', fontweight ='bold', fontsize = 15)
  plt.ylabel('Kilogramos (kg)', fontweight ='bold', fontsize = 15)
  st.pyplot(fig)


st.write("Podemos observar que a través de los años el peso de los atletas se mantiene constante para hombres y mujeres, por lo que no tiene mucho impacto en el porqué México ha estado mejorando.")
#
#
# Variación de altura a través de los años
st.write("Ahora observaremos las variaciones de la altura de los atletas hombres y mujeres:")

with st.echo():
  # M
  fig= plt.figure(figsize=(10, 5))
  sns.pointplot('Year', 'Height', data=atletas_m, palette='Set2')
  plt.title('Variación de la Altura de atletas Hombres a través del tiempo')
  plt.xlabel('Años', fontweight ='bold', fontsize = 15)
  plt.ylabel('Centímetros (cm)', fontweight ='bold', fontsize = 15)
  st.pyplot(fig)

  # F
  fig= plt.figure(figsize=(10, 5))
  sns.pointplot('Year', 'Height', data=atletas_f, palette='Set2')
  plt.title('Variación de la Altura de atletas Mujeres a través del tiempo')
  plt.xlabel('Años', fontweight ='bold', fontsize = 15)
  plt.ylabel('Centímetros (cm)', fontweight ='bold', fontsize = 15)
  st.pyplot(fig)

st.write("Igualmente podemos observar que a través de los años la altura de los atletas se mantiene constante para hombres y mujeres, por lo que no tiene mucho impacto en el porqué México ha estado mejorando.")
st.write("Esto nos dice que la razón por la que México ha estado mejorando no es la altura ni peso de los atletas, pues han sido constantes a través del tiempo. Entonces, ahora veremos la cantidad de atletas hombres y mujeres a través del tiempo:")

with st.echo():
  # M
  f_grafica = atletas_m.groupby('Year')['Sex'].value_counts()
  fig= plt.figure(figsize=(10, 5))
  plt.plot(f_grafica.loc[:,'M'])
  plt.title('Variación de atletas hombres a través del tiempo')
  plt.xlabel('Años', fontweight ='bold', fontsize = 15)
  plt.ylabel('Puntos por año', fontweight ='bold', fontsize = 15)
  st.pyplot(fig)


  # F
  f_grafica = atletas_f.groupby('Year')['Sex'].value_counts()
  fig= plt.figure(figsize=(10, 5))
  plt.plot(f_grafica.loc[:,'F'])
  plt.title('Variación de atletas mujeres a través del tiempo')
  plt.xlabel('Años', fontweight ='bold', fontsize = 15)
  plt.ylabel('Puntos por año', fontweight ='bold', fontsize = 15)
  st.pyplot(fig)

st.write("Podemos observar que el número de atletas mexicanos se ha mantenido constante, sin embargo, el número de atletas mujeres ha aumentado en gran cantidad, ahora teniendo el doble de atletas mujeres que en los 80's.")
st.write("¡Esto es un crecimiento importante!")
st.write("Podemos atribuir tentativamente al aumento del rendimiento de México en las olimpiadas al aumento de atletas mujeres a través de los años.")

st.header("Conclusiones")

st.write("Luego de haber analizado el rendimiento de México a través de los años, observamos como fueron variando las carácteristicas de los atletas y el número de atletas. Luego hicimos una predicción del rendimiento de México en los siguientes años.")
st.write("El rendimiento de México ha estado aumentando, ganando más medallas de oro, plata y bronce cada año, y esto se lo atribuimos al aumento de atletas mujeres, pues, hasta cierto año no había suficientes atletas mujeres, hasta un 'boom' en los 60's, donde el país comenzó a darse cuenta de la importancia de que las mujeres participen en este evento. Con esto, la cantidad de atletas mujeres fue aumentando y en este momento esta en su punto máximo, lo cual ha ayudado a México a mejorar.")



