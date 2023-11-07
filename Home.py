import streamlit as st
from tools.global_funcs import *

##############################################################################################################################
#Page config
st.set_page_config(
    page_title="MLB 2023 - Roberto Rascón",
    page_icon="⚾",
)
##############################################################################################################################
#Variables de sesión
set_session_variables()
##############################################################################################################################
#Sidebar
print_sidebar_template()
##############################################################################################################################

st.title("Análisis, Visualización y Modelado de datos de la temporada 2023 de la MLB")
st.subheader("Introducción")
st.markdown(
    """
    <div style='text-align: justify;'>
        <p>
            El presente proyecto describe el proceso de obtención, análisis, visualización y modelado de datos
            de la temporada 2023 (y anteriores) de Las Grandes Ligas de Béisbol (MLB, por sus siglas en Inglés,
            Major League Baseball).
        </p>
        <p>
            Se pretende con esto, demostrar que el béisbol, al estar inmerso en un mundo de variables y eventos, que a su vez
            pueden y generan una cantidad masiva de datos, es un blanco perfecto para la aplicación directa de la Ciencia de 
            Datos. La aplicación de estos métodos, aunque no exacta, es un punto de apoyo para la toma de decisiones, incluso,
            durante el desarrollo de los mismos partidos.
        </p>
        <p>
            El análisis y el trabajo realizado a los datos en esta implementación se enfocó única y exclusivamente 
            a los datos de bateo, sin embargo, es aplicable a cualquier aspecto del juego.
        </p>
        <p>
            Puede navegar por la aplicación apoyándose en los menús presentados en la barra lateral izquierda. Aunque no es
            indispensable ni necesario, se recomienda que acceda a los menús en el orden presentado.
        </p>
        <p>
            Muchas gracias, y espero sea de su agrado.
        </p>
        <span style='font-weight: bold; font-size: 20px;'>¡Bienvenido(a)!</span>
    </div>
    """,
    unsafe_allow_html=True
)


#################################################################################################################

