import streamlit as st
from datetime import date
from tools.global_funcs import *

st.set_page_config(page_title="Visualización Dinámica", page_icon="📈")
##############################################################################################################################
#Variables de sesión
set_session_variables()
#Sidebar
print_sidebar_template()
##############################################################################################################################
#from pybaseball import team_batting
#from tools.plotting_custom import plot_teams
#data = team_batting(2023)
#fig = plot_teams(data, "HR", "BB")
#st.pyplot( fig )
TOP_IND = 100 #Cuántos jugadores mostrar en el selectbox de análisis por jugador

st.header("2. Visualización Dinámica")
st.subheader("Análisis por jugador")

st.markdown(
    """
    Se le solicita al usuario que seleccione un jugador y el periodo que le gustaria visualizar de la temporada 2023 (seleccionar un periodo largo puede tomar un tiempo en graficarse
    por la cantidad de datos a evaluar). Las fuentes de datos para este propósito se obuvieron con funciones de la libreria pybaseball; y tras hacerlo se
    exportaron los datos a un CSV para posteriormente guardar los datos en una variable de sesión y con esto, evitar la llamada a función con cada "refresh" de la página:

    ```python
    if 'all_data' not in st.session_state:
        st.session_state['all_data'] = pd.read_csv( Path( st.session_state['cur_path'] , 'tools/data', 'all_data.csv') )

    data = st.session_state['all_data'].copy()
    ```
    """
)

sorted_players = st.session_state['batting_merged'].sort_values( by='WAR', ascending=False )
sorted_players = sorted_players[ 0:TOP_IND ] 
sorted_players = sorted_players.sort_values( by='Name', ascending=True )

# Obtener un diccionario para relacionar ID de jugador con Nombre
#values = sorted_players[ ["MLBID", "Name", "Team", 'team_stadium', 'team_name'] ] #Agregar todas las columnas requeridas para MAPPING
values = sorted_players.copy()
values.set_index('MLBID', inplace=True)
values = values.to_dict('index')

#parametros para date_input
min_start_date = date( 2023,3,30 )
max_end_date = date( 2023,10,1 )
default_end_date = date( 2023, 4, 5 )

players = st.sidebar.selectbox(
            'Selecciona un jugador',
            options = values.keys(),
            format_func=lambda x: f'{ values[ x ]["Name"] } ({ values[ x ]["tm"] })'
)

d = st.sidebar.date_input(
        "Seleccione periodo",
        ( min_start_date, default_end_date ),
        min_start_date,
        max_end_date,
        format="YYYY/MM/DD",
)

#https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/641355/headshot/67/current
#Desplegar informacion basica de jugador
st.sidebar.markdown(
    f"""
    <div style="text-align: center;">
        <div>
            <img src='https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/{int(players)}/headshot/67/current' width='100'/>
        </div>   
        <div>
            <p style="font-weight: bold;">{values[ players ]['Name']}</p>
            <p style="color: { values[players]['primary_color'] };background-color:{ values[players]['secondary_color'] }">{ values[ players ]['team_name'] }</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


#Obtener periodo desde INPUT
d_ini_end = format_date_collection( d )
start = d_ini_end[ 0 ] if len( d_ini_end ) > 0 else None
end = d_ini_end[ 1 ] if len( d_ini_end ) > 1 else None

if start is not None and end is not None: #Graficar unicamente si el periodo esta bien definido desde el INPUT
    st.markdown(
        """
        En los spraycharts se representan cada uno de los batazos (en el periodo definido) y dónde aterrizaron y el resultado final de la jugada.
        El spraychart general representa todos los datos en un modelo de estadio genérico, mientras que el spraychart de local representa cómo le fue al jugador en su estadio de 
        local.

        Esto está definido en python con el siguiente código:

        ```python
        def plot_spraycharts( data, selected ):
            '''
            Grafica spraychart en función de un periodo y jugador seleccionado
            '''
            #st.dataframe( data )
            fig = spraychart(data, 'generic', title='General')
            st.pyplot( fig )

            sub_data = data[ data['home_team'] == selected['code_plot'] ]
            fig = spraychart(sub_data, selected['team_stadium'], title=f"De local (estadio de { selected['team_name'] })")
            st.pyplot( fig )
        ```
        """
    )
    #to_csv = statcast("2023-03-30","2023-10-01")
    #to_csv.to_csv("tools/data/all_data.csv")
    data = st.session_state["all_data"].copy()
    data = data[ data["batter"] == players ]
    data["game_date"] = pd.to_datetime( data["game_date"] )
    condition = ( data["game_date"] >= start ) & ( data["game_date"] <= end )
    data = data.loc[ condition ]
    data = data[ ~data["events"].isnull() ]
    data["is_hit"] = np.where( data["events"].isin( [ "single","double","triple","home_run" ] ), True, False)
    data["is_at_bat"] = np.where( data["events"].isin( [ "sac_fly","sac_bunt","hit_by_pitch","walk" ] ), False, True)

    #data = statcast_batter( start, end, players )
    plot_spraycharts( data, values[ players ] ) #field

    st.markdown(
        """
        Se hace uso de heatmaps para representar la zona de strike dividida en 9 ubicaciones, cada número dentro de la matriz representa la cantidad
        de hits o el promedio de bateo del bateador seleccionado al hacer contacto con la pelota dentro de esa área (desde la perspectiva del receptor).

        Todo: aplicado a todos los datos  \n
        VS L: resultado ante lanzadores zurdos  \n
        VS R: resultado ante lanzadores derechos  \n

        El código detras de cada heatmap es el siguiente:

        ```python
        def plot_single_heatmap( batting_data, title, column_name="pitch_type", max = None ):
            '''
            Grafica un heatmap en función de data y title. Se asume que los datos provienen de statcast( start, end )
            '''
            #Combinar hits_heatmap con un nuevo dataframe con ceros para tener valores para todas las zonas
            zeros = {'zone': range(1,14), column_name: [0] * 13}
            df = pd.DataFrame( zeros )
            df.set_index('zone', inplace=True)
            df.reset_index()

            combined_series = pd.concat( [ batting_data, df] )
            unique_series = combined_series[ ~combined_series.index.duplicated(keep='first') ]

            #Inicializar Z
            z = np.array([
                [0,0,0],
                [0,0,0],
                [0,0,0]
            ], dtype="float")

            for i in range( 3 ):
                for j in range( 3 ):
                    index_value = 1 + j + 3 * i
                    z[ i ][ j ] = unique_series.loc[ index_value ][ column_name ]
                    #st.write( [ i, j, index_value ] )

            fig = px.imshow( z, title=title, text_auto=True, color_continuous_scale = 'rdylbu', range_color = [ z.min(), max or z.max() ], x = ['A','B','C'], y = ['Y1 ','Y2 ','Y3 '])

            col1, col2 = st.columns(2)
            with col1: st.write(unique_series)
            with col2: st.plotly_chart( fig )
            #fig.update_traces(showlegend=False)
            
            #st.write( px.colors.named_colorscales() )
        ```

        """
    )
    plot_heatmaps( data ) #Strike zone

    st.markdown(
        """
        Hits acumulados muestra el acumulado de hits para el jugador a lo largo del tiempo.
        La gráfica se apoya de la siguiente lógica:

        ```python
        def plot_stat_acum( data, title ):
            st.subheader( title )
            data = pd.DataFrame( data )
            st.line_chart( data )

        def plot_acums( data ):
            #Todos los AB válidos son diferentes de None, aquí solo estamos considerando los hits
            all_data_hits = data[ data["is_hit"] == True ]
            hits_acum = all_data_hits.groupby('game_date')['pitch_type'].count().cumsum() #contar agrupando por fecha y luego sumar
            plot_stat_acum( hits_acum, "Hits acumulados" )

        data = data[ data["batter"] == players ]
        data["game_date"] = pd.to_datetime( data["game_date"] )
        condition = ( data["game_date"] >= start ) & ( data["game_date"] <= end )
        data = data.loc[ condition ]
        data = data[ ~data["events"].isnull() ]
        data["is_hit"] = np.where( data["events"].isin( [ "single","double","triple","home_run" ] ), True, False)
        data["is_at_bat"] = np.where( data["events"].isin( [ "sac_fly","sac_bunt","hit_by_pitch","walk" ] ), False, True)
        
        plot_acums( data )
        ```
        Se utiliza la función para preparar gráficas similares de otras estadísticas modificando el filtro (single, double, triple o home_run) antes de hacer la agrupación
        """
    )
    plot_acums( data )

###############################################################################################################################################