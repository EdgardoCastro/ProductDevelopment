import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import numpy as np
import warnings
import plotly.express as px
import base64


# Configuración de página
st.set_page_config(page_title="Curso Product Development", page_icon=":bar_chart:")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()    
    return base64.b64encode(data).decode()
def set_background(png_file):
    bin_str = get_base64(png_file)    
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('C:/Users/PC2/Documents/Maestría BIA/Año 2/T4/ProductDevelopment/Proyecto1/imagen1.png')

# Opción para deshabilitar la advertencia
st.set_option('deprecation.showPyplotGlobalUse', False)

# Sidebar
st.sidebar.markdown("# Navegación")
opcion = st.sidebar.selectbox("Selecciona una pestaña", ["Carga de datos","Exploración de datos"])

# Obtener o crear la sesión
session_state = st.session_state

# Inicializar datos si no existe en la sesión
if "datos" not in session_state:
    session_state.datos = None

# Contenido de la pestaña "Exploración de datos"
if opcion == "Carga de datos":
    col1, col2 = st.columns([2, 1])

    col1.markdown("# APLICACIÓN: CURSO PRODUCT DEVELOPMENT")
    col1.markdown("Carga y exploración de datos")

    datos = col2.file_uploader("Carga tu dataset en formato .xlsx")

    st.title("Tu dataset")

    # Actualizar datos en la sesión
    if datos is not None:
        session_state.datos = pd.read_excel(datos)

    if session_state.datos is not None:
        st.write(session_state.datos)
        
        # Botón para borrar el dataset
        if st.button("Borrar Dataset"):
            session_state.datos = None
            st.success("Dataset borrado exitosamente.")
    else:
        st.info("Por favor, carga un archivo .xlsx")

# Contenido de la otra pestaña
elif opcion == "Exploración de datos":
    st.title("Exploración estadística de los datos")
    st.write("Agrega aquí el contenido de tu segunda pestaña.")

    # Verifica si los datos han sido cargados
    if session_state.datos is not None:
        df = session_state.datos

        categoricas = []
        continuas = []
        discretas = []

        for colName in df.columns:
            if df[colName].dtype == 'O':
                categoricas.append(colName)
            else:
                unique_values = df[colName].nunique()
                if (df[colName].dtype == 'int64' or df[colName].dtype == 'float64'):
                    if unique_values <= 30:
                        discretas.append(colName)
                    else:
                        continuas.append(colName)

        st.info("Variables Categóricas:")
        col1, col2 = st.columns(2)
        col1.write("VARIABLE")
        col2.write("EJEMPLO")
        for col in categoricas:
            ejemplo_valor = df[col].iloc[0] if not df[col].empty else None
            col1.write(col)
            col2.write(ejemplo_valor)

        st.info("Variables Numéricas Continuas:")
        col1, col2 = st.columns(2)
        col1.write("VARIABLE")
        col2.write("EJEMPLO")
        for col in continuas:
            ejemplo_valor = df[col].iloc[0] if not df[col].empty else None
            col1.write(col)
            col2.write(ejemplo_valor)

        st.info("Variables Numéricas Discretas:")
        col1, col2 = st.columns(2)
        col1.write("VARIABLE")
        col2.write("EJEMPLO")
        for col in discretas:
            ejemplo_valor = df[col].iloc[0] if not df[col].empty else None
            col1.write(col)
            col2.write(ejemplo_valor)

    selected_variable = st.selectbox("Selecciona una variable numérica continua:", continuas)

    if selected_variable:
            # Graficar densidad
            st.subheader(f"Gráfica de Densidad para '{selected_variable}'")
            plt.figure(figsize=(8, 6))
            sns.kdeplot(df[selected_variable], fill=True)
            plt.xlabel(selected_variable)
            plt.ylabel("Densidad")

            # Calcular estadísticas
            media = df[selected_variable].mean()
            mediana = df[selected_variable].median()
            moda = df[selected_variable].mode().values[0]

            # Agregar líneas para media y mediana
            plt.axvline(media, color='red', linestyle='--', label=f'Media: {media:.2f}')
            plt.axvline(mediana, color='green', linestyle='--', label=f'Mediana: {mediana:.2f}')

            # Calcular y agregar líneas para desviación estándar y varianza
            std_dev = df[selected_variable].std()
            variance = df[selected_variable].var()
            plt.axvline(media + std_dev, color='orange', linestyle='--', label=f'Desviación Estándar: {std_dev:.2f}')
            plt.axvline(media - std_dev, color='orange', linestyle='--')
            plt.axvline(media + 2*std_dev, color='purple', linestyle='--', label=f'Varianza: {variance:.2f}')
            plt.axvline(media - 2*std_dev, color='purple', linestyle='--')

            # Mostrar leyenda en la gráfica
            plt.legend()

            st.pyplot()

    selected_variable = st.selectbox("Selecciona una variable numérica discreta:", discretas)
    if selected_variable:
 # Graficar histograma
            st.subheader(f"Histograma para '{selected_variable}'")
            plt.figure(figsize=(8, 6))
            sns.histplot(df[selected_variable], kde=False)
            plt.xlabel(selected_variable)
            plt.ylabel("Frecuencia")

              # Calcular estadísticas
            media = df[selected_variable].mean()
            mediana = df[selected_variable].median()
            moda = df[selected_variable].mode().values[0]

            # Agregar líneas para media, mediana y moda
            plt.axvline(media, color='red', linestyle='--', label=f'Media: {media:.2f}')
            plt.axvline(mediana, color='green', linestyle='--', label=f'Mediana: {mediana:.2f}')
            plt.axvline(moda, color='blue', linestyle='--', label=f'Moda: {moda:.2f}')

            # Calcular y agregar líneas para desviación estándar y varianza
            std_dev = df[selected_variable].std()
            variance = df[selected_variable].var()
            plt.axvline(media + std_dev, color='orange', linestyle='--', label=f'Desviación Estándar: {std_dev:.2f}')
            plt.axvline(media - std_dev, color='orange', linestyle='--')
            plt.axvline(media + 2*std_dev, color='purple', linestyle='--', label=f'Varianza: {variance:.2f}')
            plt.axvline(media - 2*std_dev, color='purple', linestyle='--')

            # Mostrar leyenda en la gráfica
            plt.legend()

            st.pyplot()

    selected_variable = st.selectbox("Selecciona una variable categorica:", categoricas)
    if selected_variable:
            
            # Gráfica de barras
            st.subheader(f"Gráfica de Barras para '{col}'")
            plt.figure(figsize=(8, 6))
            sns.countplot(data=df, x=col)
            plt.xlabel(col)
            plt.ylabel("Total")
            plt.xticks(rotation=45)
            st.pyplot()

    else:
            st.warning("Por favor, selecciona una variable numérica continua antes de generar la gráfica.")


    # Agregar dos selectbox para seleccionar las variables a comparar
    selected_variable_x = st.selectbox("Selecciona una variable numérica continua (X):", continuas + discretas)
    selected_variable_y = st.selectbox("Selecciona otra variable numérica continua (Y):", continuas + discretas)

    if selected_variable_x and selected_variable_y:
          # Crear el scatter plot
        st.subheader("Gráfico de Dispersión")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=selected_variable_x, y=selected_variable_y)
        plt.xlabel(selected_variable_x)
        plt.ylabel(selected_variable_y)

    # Calcular el coeficiente de correlación
        corr_coeff = df[selected_variable_x].corr(df[selected_variable_y])
        st.write(f"Coeficiente de Correlación (Pearson): {corr_coeff:.2f}")

        st.pyplot()
    else:
        st.warning("Por favor, selecciona dos variables numéricas continuas antes de generar el gráfico de dispersión y calcular el coeficiente de correlación.")


    # Agregar dos selectboxes para seleccionar las variables a comparar en los ejes X e Y
    selected_variable_x = st.selectbox("Selecciona una variable para el eje X:", continuas + discretas)
    selected_variable_y = st.selectbox("Selecciona una variable de fecha:", df.columns[df.dtypes == 'datetime64[ns]'])

    if selected_variable_x and selected_variable_y:
        st.subheader("Gráfico de Serie Temporal")
        
        if selected_variable_x in continuas:
            # Sumar las variables continuas por fecha
            df_grouped = df.groupby(selected_variable_y)[selected_variable_x].sum().reset_index()
            fig = px.line(df_grouped, x=selected_variable_y, y=selected_variable_x, title=f'Serie Temporal (Suma) de {selected_variable_x}')
        elif selected_variable_x in discretas:
            # Calcular el promedio de las variables discretas por fecha
            df_grouped = df.groupby(selected_variable_y)[selected_variable_x].mean().reset_index()
            fig = px.line(df_grouped, x=selected_variable_y, y=selected_variable_x, title=f'Serie Temporal (Promedio) de {selected_variable_x}')
        
        st.plotly_chart(fig)
    else:
        st.warning("Por favor, selecciona una variable para el eje X y una variable de fecha antes de generar el gráfico de serie temporal.")

    # Agregar dos selectboxes para seleccionar la variable categórica en el eje X y la variable continua en el eje Y
    selected_variable_x = st.selectbox("Selecciona una variable categórica para el eje X:", categoricas)
    selected_variable_y = st.selectbox("Selecciona una variable continua para el eje Y:", continuas)

    if selected_variable_x and selected_variable_y:
        st.subheader("Gráfico de Boxplot")
        
        # Crear el boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=selected_variable_x, y=selected_variable_y)
        plt.xlabel(selected_variable_x)
        plt.ylabel(selected_variable_y)
        plt.title(f'Boxplot de {selected_variable_y} por {selected_variable_x}')
        
        st.pyplot()
    else:
        st.warning("Por favor, selecciona una variable categórica para el eje X y una variable continua para el eje Y antes de generar el gráfico de boxplot.")
    

    # Agregar dos selectboxes para seleccionar las variables categóricas a comparar
    selected_variable_x = st.selectbox("Selecciona una variable categórica para el eje X:", categoricas, key="selectbox_x")
    selected_variable_y = st.selectbox("Selecciona otra variable categórica para el eje Y:", categoricas, key="selectbox_y")


    if selected_variable_x and selected_variable_y:
        st.subheader("Gráfica de Mosaico con Coeficiente de Cramer")
        
        # Crear la tabla de contingencia
        contingency_table = pd.crosstab(df[selected_variable_x], df[selected_variable_y])
        
        # Calcular el coeficiente de contingencia de Cramer
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        cramers_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        
        # Crear la gráfica de mosaico
        plt.figure(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
        plt.xlabel(selected_variable_y)
        plt.ylabel(selected_variable_x)
        plt.title(f'Gráfica de Mosaico\nCoeficiente de Cramer: {cramers_v:.2f}')
        
        st.pyplot()
    else:
        st.warning("Por favor, selecciona dos variables categóricas antes de generar la gráfica de mosaico y calcular el coeficiente de contingencia de Cramer.")




else:
        st.warning("Por favor, carga un archivo .xlsx antes de explorar los datos.")



