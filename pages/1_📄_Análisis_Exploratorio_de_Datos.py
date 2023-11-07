import streamlit as st
import altair as alt
from tools.global_funcs import *

st.set_page_config(page_title="Análisis Exploratorio de Datos", page_icon="📄")
##############################################################################################################################
#Variables de sesión
set_session_variables()
##############################################################################################################################
#Sidebar
print_sidebar_template()
##############################################################################################################################
#Intro section
st.header("1. Análisis Exploratorio de Datos")
st.subheader("Obtención y limpieza de datos")
# Load data
st.markdown(
    """
    Se utiliza la libreria pybaseball para cargar los datos de bateo de todos los jugadores clasificados
    de la temporada 2023
    ```python
    from pybaseball import  batting_stats

    if 'cur_path' not in st.session_state:
        st.session_state['cur_path'] = Path(__file__).resolve().parent.parent
    if 'start' not in st.session_state:
        st.session_state['start'] = 2023
    if 'end' not in st.session_state:
        st.session_state['end'] = 2023
    if 'batting' not in st.session_state:
        batting_filepath = Path( st.session_state['cur_path'] , 'tools/data', 'batting_2023.csv') 
        if os.path.exists( batting_filepath ):
            batting = pd.read_csv( batting_filepath, index_col=0 )
        else:
            batting = batting_stats( st.session_state['start'], st.session_state['end'], qual=502 ) #502 es el número de turnos legales que un bateador debe tener al concluir la temporada para ser considerado en los lideres a la ofensiva
            batting.to_csv( batting_filepath )
    ```
    
    Se aplica una limpieza en los datos para considerar solamente las columnas en la que todos sus valores
    no son nulos

    ```python
        #Limpieza de datos
        null_count = batting.isnull().sum()
        complete_cols = list( batting.columns[ null_count == 0 ] ) #Solo las columnas que no tienen vacíos
        batting = batting[ complete_cols ]
        st.session_state['batting'] = batting.copy() #Guardar dataframe en variable de sesion
    ```

    Se obtiene el siguiente DataFrame (limitado a 10 registros en esta visualización)
"""
)

batting = st.session_state['batting'].copy()
st.dataframe( batting.head( 10 ), hide_index=True )

st.subheader("Dataframe de equipos")
st.markdown(
    """
    Se cargan los datos de todos los equipos de MLB en un nuevo DataFrame con un origen de datos diferente al primer dataframe,
    se modifica una línea para cambiar el valor FLA a MIA del equipo Miami Marlins,
    ya que este es el valor que necesitamos para hacer un JOIN con el DataFrame de estadísticas de bateo.
    Todos los demás equipos coinciden.
    
    ```python
    if 'mlb_teams' not in st.session_state:
        url = "https://raw.githubusercontent.com/glamp/mlb-payroll-and-wins/master/team-colors.csv"
        mlb_teams_path = Path( st.session_state['cur_path'] , 'tools/data', 'mlb_teams.csv') 
        if os.path.exists( mlb_teams_path ):
            mlb_teams = pd.read_csv( mlb_teams_path )     
        else:
            mlb_teams = pd.read_csv( url )
            mlb_teams.to_csv( mlb_teams_path )
        mlb_teams.loc[ mlb_teams[ 'team_name' ] == 'Miami Marlins', 'tm' ] = 'MIA'
        st.session_state['mlb_teams'] = mlb_teams.copy()
    ```
    
    Los datos de este DataFrame son los siguientes:
    """
)

mlb_teams = st.session_state['mlb_teams']
st.dataframe( mlb_teams, hide_index=True )

st.subheader("Operacion INNER JOIN entre dataframes")
st.markdown(
    """
    Se hace el JOIN entre los dos dataframes, considerando de mlb_teams únicamente las columnas tm y league

    ```python
    #con INNER JOIN eliminamos aquellos jugadores que no tienen equipo asignado
    st.session_state['batting_merged'] = st.session_state['batting'].merge( st.session_state['mlb_teams'][ ["tm", "team_name","league", "primary_color","secondary_color"] ], left_on="Team", right_on="tm", how="inner")
    ```

    Se hace un JOIN más con un nuevo dataframe (sfd) que contiene todos los IDs del jugador en distintas plataformas especializadas en béisbol.
    De esta forma podemos hacer uso de APIs y funciones en Pybaseball de manera dinámica

    ```python
    #Segundo JOIN SFBB (Smart Fantasy Baseball Source https://www.smartfantasybaseball.com/tools/)
    #Necesario para obtener los IDs del jugador en distintas plataformas especializadas en béisbol
    sfd = pd.read_csv( Path( st.session_state['cur_path'] , 'tools/data', 'SFBB Player ID Map - PLAYERIDMAP.csv') )
    sfd = sfd[ sfd['POS'] != "P" ] #Ignorar registros de pitchers
    st.session_state['batting_merged']['IDfg'] = st.session_state['batting_merged']['IDfg'].astype( str ) #Se debe convertir a string antes de poder hacer el JOIN
    sfd['IDFANGRAPHS'] = sfd['IDFANGRAPHS'].astype( str ) #Se debe convertir a string antes de poder hacer el JOIN
    st.session_state['batting_merged'] = st.session_state['batting_merged'].merge( sfd, left_on="IDfg", right_on="IDFANGRAPHS", how="left" )

    ```

    Un último JOIN que permite obtener el nombre de equipo, tal y como necesita la función spraychart (incluida en la sección de visualización dinámica de datos)

    ```python
    #Tercer JOIN para la información de ESTADIO
    stadiums = pd.read_csv( Path( st.session_state['cur_path'], 'tools/data', 'mlbstadiums_custom.csv') )
    st.session_state['batting_merged'] = st.session_state['batting_merged'].merge( stadiums, left_on="Team", right_on="code", how="left" )
    ```

    De esta forma tenemos un solo DataFrame que incluye todas las columnas necesarias; se concluye con la limpieza de datos.
    El DataFrame resultante es el siguiente (limitado a 10 registros por visualización):
    """
)

#con INNER JOIN eliminamos aquellos jugadores que no tienen equipo asignado
batting_merged = st.session_state['batting_merged'].copy()

st.dataframe( batting_merged.head(10), hide_index=True )

st.header("Gráficos variados con \"input\" de usuario")

st.markdown(
    """
    <div style="text-align: justify;">
        <p>
            Se presenta una visualización de datos determinada por inputs de usuario con base en el dataframe resultante,
            en este caso se le solicita al usuario que seleccione una estadística y el orden que le gustaria desplegar los datos.
            El despliegue de datos se separa como un TOP 10 para jugadores de la Liga Americana y la Liga Nacional.
        </p>
    </div>
    
    """,
    unsafe_allow_html=True
)

#Muestra los inputs en dos columnas (similar a bootstrap)
col1, col2 = st.columns(2)

with col1:
    stats = st.selectbox(
        'Selecciona una estadística',
        batting.columns.tolist()[5:]
    )

with col2:
    sort_order = st.selectbox(
        'Orden',
        ('DESC','ASC')
    )

sort_selected = sort_order == 'ASC'
TOP_VAL = 10

#AMERICAN LEAGUE
top_al = batting_merged [ batting_merged["league"] == "AL" ]
top_al = top_al.sort_values( by=stats, ascending=sort_selected )
top_al = top_al.head(TOP_VAL)
top_al = top_al[ [ 'Name', 'Team', stats ] ]
#NATIONAL LEAGUE
top_nl = batting_merged [ batting_merged["league"] == "NL" ]
top_nl = top_nl.sort_values( by=stats, ascending=sort_selected )
top_nl = top_nl.head(TOP_VAL)
top_nl = top_nl[ [ 'Name', 'Team', stats ] ]

st.subheader("AMERICAN LEAGUE")
tab1, tab2 = st.tabs([ "Grafica", "Tabla"] ) # Desplegar como tabs

with tab1:
    c = ( alt.Chart(top_al).mark_bar().encode( alt.X( 'Name', sort='-y'), alt.Y( stats) ) )
    st.altair_chart( c, use_container_width=True )
with tab2:
    st.dataframe(top_al, hide_index=True)

st.subheader("NATIONAL LEAGUE")
tab1, tab2 = st.tabs([ "Grafica", "Tabla"] ) # Desplegar como tabs
with tab1:
    c = ( alt.Chart(top_nl).mark_bar().encode( alt.X( 'Name', sort='-y'), alt.Y( stats) ) )
    st.altair_chart( c, use_container_width=True )
with tab2:
    st.dataframe(top_nl, hide_index=True)
