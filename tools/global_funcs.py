from typing import Optional
import os
#from datetime import datetime
#from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
#import altair as alt
from pathlib import Path
from pybaseball import  batting_stats
#from tools.plotting_custom import plot_strike_zone
#from pybaseball import statcast
import plotly.express as px
#from pybaseball import statcast_batter
from tools.plotting_custom import spraychart

def print_sidebar_template():
    """
    Imprime información constante en la vista, debe llamarse en primera instancia en todas las vistas
    """
    st.sidebar.title("Código Facilito: Proyecto final de Bootcamp de Ciencia de Datos 2023")
    st.sidebar.markdown(
        """
        <div>
            <span>Presenta</span><br>
            <span>Roberto Rascón Meza</span>
        </div>
        """
        , unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<hr>',
        unsafe_allow_html=True
    )

def set_session_variables():
    """
    Define y establece las variables de sesión, debe llamarse en cada "vista" como si de un constructor se tratase.
    Las variables se establecen únicamente si no tienen valor en la sesión activa
    """
    if 'cur_path' not in st.session_state:
        st.session_state['cur_path'] = Path(__file__).resolve().parent.parent
    if 'start' not in st.session_state:
        st.session_state['start'] = 2023
    if 'end' not in st.session_state:
        st.session_state['end'] = 2023
    if 'all_data' not in st.session_state:
        st.session_state['all_data'] = pd.read_csv( Path( st.session_state['cur_path'] , 'tools/data', 'all_data.csv') )
    if 'batting' not in st.session_state:
        batting_filepath = Path( st.session_state['cur_path'] , 'tools/data', 'batting_2023.csv') 
        if os.path.exists( batting_filepath ):
            batting = pd.read_csv( batting_filepath, index_col=0 )
        else:
            batting = batting_stats( st.session_state['start'], st.session_state['end'], qual=502 ) #502 es el número de turnos legales que un bateador debe tener al concluir la temporada para ser considerado en los lideres a la ofensiva
            batting.to_csv( batting_filepath )
        # Limpieza de datos
        null_count = batting.isnull().sum()
        complete_cols = list( batting.columns[ null_count == 0 ] ) #Solo las columnas que no tienen vacíos
        batting = batting[ complete_cols ]
        st.session_state['batting'] = batting.copy()
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
    if 'batting_merged' not in st.session_state:
        st.session_state['batting_merged'] = st.session_state['batting'].merge( st.session_state['mlb_teams'][ ["tm", "team_name","league", "primary_color","secondary_color"] ], left_on="Team", right_on="tm", how="inner")
        #Segundo JOIN SFBB (Smart Fantasy Baseball Source https://www.smartfantasybaseball.com/tools/)
        #Necesario para obtener los IDs del jugador en distintas plataformas especializadas en beisbol
        sfd = pd.read_csv( Path( st.session_state['cur_path'] , 'tools/data', 'SFBB Player ID Map - PLAYERIDMAP.csv') )
        sfd = sfd[ sfd['POS'] != "P" ] #Ignorar registros de pitchers 
        st.session_state['batting_merged']['IDfg'] = st.session_state['batting_merged']['IDfg'].astype( str ) #Se debe convertir a string antes de poder hacer el JOIN
        sfd['IDFANGRAPHS'] = sfd['IDFANGRAPHS'].astype( str ) #Se debe convertir a string antes de poder hacer el JOIN
        st.session_state['batting_merged'] = st.session_state['batting_merged'].merge( sfd, left_on="IDfg", right_on="IDFANGRAPHS", how="left" )
        #Tercer JOIN para la informacion de ESTADIO
        stadiums = pd.read_csv( Path( st.session_state['cur_path'], 'tools/data', 'mlbstadiums_custom.csv') )
        st.session_state['batting_merged'] = st.session_state['batting_merged'].merge( stadiums, left_on="Team", right_on="code", how="left" )
    if 'batting_ml' not in st.session_state:
        batting_filepath = Path( st.session_state['cur_path'] , 'tools/data', 'batting.csv') 
        if os.path.exists( batting_filepath ):
            batting = pd.read_csv( batting_filepath, index_col=0 )
        else:
            batting = batting_stats( 2002, 2023, qual=502 )
            batting.to_csv( batting_filepath )

        batting = batting[ batting["Season"] != 2020 ] #Esta temporada fue reducida por restricciones COVID19
        st.session_state['batting_ml'] = batting

def format_date_collection( dates: tuple ) -> list:
    """
    Convierte fechas de una tupla y devuelve la colección entera en una lista
    """
    some_list = []
    for r in dates:
        some_list.append( pd.to_datetime( r ) )
    
    return some_list

def count_true( x ) -> int:
    """
    Función llamada por el método agg en un dataframe tras el groupby, para retornar la cantidad de elementos que cumplen la condición True en la columna especificada
    """
    return x.sum()

def reset_dataframe( data ):
    """
    Devuelve el dataframe tras hacer un conteo agrupando por "zone"
    """
    df = data.groupby(['zone'])['pitch_type'].count()
    df = pd.DataFrame( df )
    return df

def get_dataframe_for_avg( df ):
    """
    Retorna el dataframe con una columna extra de avg (average) y remueve las columnas no necesarias para graficar el heatmap
    """
    df = df.groupby('zone').agg({'is_hit': count_true, 'is_at_bat': count_true})
    df = df.rename(columns={'is_hit': 'count_hit', 'is_at_bat': 'count_at_bat'})
    df['avg'] = df['count_hit']/( df['count_at_bat'] )

    return df.drop( columns=['count_hit','count_at_bat']  )

def plot_spraycharts( data, selected ):
    """
    Grafica los spraycharts en función de un periodo y jugador seleccionado
    """
    #st.dataframe( data )
    fig = spraychart(data, 'generic', title='General')
    st.pyplot( fig )
    sub_data = data[ data['home_team'] == selected['code_plot'] ]
    fig = spraychart(sub_data, selected['team_stadium'], title=f"De local (estadio de { selected['team_name'] })")
    st.pyplot( fig )

def plot_single_heatmap( batting_data, title, column_name="pitch_type", max = None ):
    """
    Grafica un heatmap en función de data y title.
    Se asume que los datos provienen de statcast( start, end )
    """
    #Combinar hits_heatmap con un nuevo dataframe con ceros para tener valores para todas las zonas
    zeros = {'zone': range(1,14), column_name: [0] * 13}
    df = pd.DataFrame( zeros )
    df.set_index('zone', inplace=True)
    df.reset_index()

    combined_series = pd.concat( [ batting_data, df ] )
    unique_series = combined_series[ ~combined_series.index.duplicated( keep='first' ) ]

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

def get_acum_hit( data, hit ):
    """
    Realiza una agrupación por fecha de la estadística "hit" y retorna el conteo
    """
    hit_acum = data[ data["events"] == hit ]
    hit_acum = hit_acum.groupby('game_date')['pitch_type'].count().cumsum() #contar agrupando por fecha y luego sumar
    return hit_acum

def plot_stat_acum( data, title ):
    """
    Grafica el acumulado definido en data usando un line_chat de streamlit
    """
    st.subheader( title )
    data = pd.DataFrame( data )
    st.line_chart( data )

def plot_heatmaps( data ):
    """
    Grafica distinta variedad de heatmaps
    """
    #Todos los AB validos son diferentes de None, aqui solo estamos considerando los hits
    all_data_hits = data[ data["is_hit"] == True ]
    all_data_hits['game_date'] = pd.to_datetime( all_data_hits['game_date'] ) #Convertir a fecha
    left_pitcher_data_hits = all_data_hits[ all_data_hits['p_throws'] == "L" ]
    right_pitcher_data_hits = all_data_hits[ all_data_hits['p_throws'] == "R" ]

    st.markdown("Referencia de strikezone: https://www.researchgate.net/figure/Game-Day-Zones-as-defined-by-Statcast-and-Baseball-Savant-The-strike-zone-is-presented_fig1_358572353")

    st.subheader("Heatmap de hits")
    tab1, tab2, tab3 = st.tabs([ "Todo", "VS L", "VS R"] ) # Desplegar como tabs
    with tab1: plot_single_heatmap( reset_dataframe( all_data_hits ), "Todo" )
    with tab2: plot_single_heatmap( reset_dataframe( left_pitcher_data_hits ), "VS L")
    with tab3: plot_single_heatmap( reset_dataframe( right_pitcher_data_hits ), "VS R")

    all_data_average = data.copy()
    left_pitcher_average = all_data_average[ all_data_average['p_throws' ] == "L" ]
    right_pitcher_average = all_data_average[ all_data_average['p_throws'] == "R" ]

    st.subheader("Heatmap de AVG")
    tab1, tab2, tab3 = st.tabs([ "Todo", "VS L", "VS R"] ) # Desplegar como tabs
    with tab1: plot_single_heatmap( get_dataframe_for_avg( all_data_average ), "Todo", "avg", .500)
    with tab2: plot_single_heatmap( get_dataframe_for_avg( left_pitcher_average ), "VS L", "avg", .500)
    with tab3: plot_single_heatmap( get_dataframe_for_avg( right_pitcher_average ), "VS R", "avg", .500)

def plot_acums( data ):
    """
    Grafica los acumulados de Hits, Single, Double, Triple y HR
    """
    #Todos los AB validos son diferentes de None, aqui solo estamos considerando los hits
    all_data_hits = data[ data["is_hit"] == True ]
    hits_acum = all_data_hits.groupby('game_date')['pitch_type'].count().cumsum() #contar agrupando por fecha y luego sumar
    
    plot_stat_acum( hits_acum, "Hits totales acumulados" )
    
    col1, col2 = st.columns(2)
    with col1:
        plot_stat_acum( get_acum_hit( all_data_hits, "single" ), "Sencillos Acumulado")
        plot_stat_acum( get_acum_hit( all_data_hits, "double" ), "Dobles Acumulado")
    with col2:
        plot_stat_acum( get_acum_hit( all_data_hits, "triple" ), "Triples Acumulado")
        plot_stat_acum( get_acum_hit( all_data_hits, "home_run" ), "HR Acumulado")