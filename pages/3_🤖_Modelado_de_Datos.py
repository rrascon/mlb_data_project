import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from pybaseball import batting_stats
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tools.global_funcs import *

st.set_page_config(page_title="Modelado de Datos", page_icon="🤖")
##############################################################################################################################
#Variables de sesión
set_session_variables()
##############################################################################################################################
#Sidebar
print_sidebar_template()
##############################################################################################################################

def next_season(player):
    """
    Retorna el STAT de la siguiente temporada (si es que hay) del registro que accede
    """
    player = player.sort_values("Season")
    player[ f"Next_{ STAT_COL }" ] = player[ STAT_COL ].shift(-1)
    return player

def backtest( data, model, predictors, start = 10, step = 1 ):
    """
    Realiza un backtesting para predecir valores utilizando las temporadas anteriores como datos históricos
    Backtesting: es un término que se usa en modelado para referirse a probar un modelo predictivo utilizando datos históricos
    """
    all_predictions = []
    years = sorted( data["Season"].unique() )

    #Entrena al modelo con todos los años disponibles
    for i in range( start, len( years ), step ):

        current_year = years[ i ]
        train = data[ data["Season"] < current_year ] #train data = Años anteriores al año del registro seleccionado
        test = data[ data["Season"] == current_year ] #test data = Datos de año de registro seleccionado
        
        model.fit( train[ predictors ], train[ f"Next_{ STAT_COL }" ] ) #Training data, target values
        
        preds = model.predict( test[ predictors ] ) #Samples para predecir
        preds = pd.Series( preds, index=test.index )
        combined = pd.concat( [ test[ f"Next_{ STAT_COL }" ], preds ], axis = 1 ) #Agrega la columna de "predicción" al dataframe
        combined.columns = ["actual", "prediction"] #Solo para renombrar columnas
        
        all_predictions.append( combined ) #Guarda subset en lista

    return pd.concat( all_predictions ) #Concatena todos los subset en uno solo

def group_averages( df ):
    """
    Comparación entre valor de STAT para el registro que accede y la media
    """
    return df[ STAT_COL ] / df[ STAT_COL ].mean()

def player_history(df):
    """
    Agrega las columnas STAT_corr = Correlación y STAT_diff = diferencia entre valor y valor anterior (de la temporada pasada)
    """
    df = df.sort_values("Season")
        
    df[ "player_season" ] = range( 0, df.shape[ 0 ] ) #Valores de 0 a N = Número de registros relacionados a este jugador (temporadas)

    expanded = df[ [ "player_season", STAT_COL ] ].expanding().corr() #Tabla de correlación expandida entre player_season y STAT
    reduced = expanded.loc[ ( slice( None ), "player_season" ), STAT_COL ] #Todos los registros de player_season con columnas player_season y STAT
    df[ f"{ STAT_COL }_corr" ] = list( reduced )

    #Calcula la correlación entre player_season y STAT
    df[ f"{ STAT_COL }_corr" ].fillna(0, inplace=True) #Rellena NaN con ceros
    
    df[ f"{ STAT_COL }_diff" ] = df[ STAT_COL ] / ( df[ STAT_COL ].shift(1) ) #Comparación entre valor de STAT entre fila y fila anterior
    df[ f"{ STAT_COL }_diff" ].fillna(1, inplace=True) 
    df[ f"{ STAT_COL }_diff" ][ df[ f"{ STAT_COL }_diff" ] == np.inf ] = 1 #Convertir resultado de división entre 0 a 1 (1 es valor máximo)
    
    return df

st.header("3. Modelado de Datos")
st.subheader("Descripción")

st.markdown(
    """
    Se implementa el Modelado de Datos con el propósito de tratar de predecir estadísticas y comparar resultados en base a los datos
    de bateo de todos los jugadores en MLB en los últimos 20 años. En la barra lateral izquierda, el selectbox le permite 
    seleccionar una estadística a evaluar; el mismo modelo y proceso se aplica a cada estadística con resultados variables
    """
)

st.subheader("Procedimiento de implementación")

st.markdown(
    """
    El proceso inicia al cargar los datos de bateo de las últimas 20 temporadas. Lo anterior se apoya de la siguiente
    lógica en Python:

    ```python
    if 'batting_ml' not in st.session_state:
        batting_filepath = Path( st.session_state['cur_path'] , 'tools/data', 'batting.csv') 
        if os.path.exists( batting_filepath ):
            batting = pd.read_csv( batting_filepath, index_col=0 )
        else:
            batting = batting_stats( 2002, 2023, qual=502 )
            batting.to_csv( batting_filepath )

        batting = batting[ batting["Season"] != 2020 ] #Esta temporada fue reducida por restricciones COVID19
        st.session_state['batting_ml'] = batting

    batting = st.session_state["batting_ml"].copy() #Obtener valor desde sesión y evitar nuevamente la carga de datos
    batting = batting.groupby("IDfg", group_keys=False).filter( lambda x: x.shape[0] > 2 ) #Solamente jugadores con más de dos temporadas registradas
    ```

    El dataframe resultante es el siguiente:
    """
)

selected_stat = st.sidebar.selectbox(
    'Selecciona una estadística',
    options = [ "WAR", "AB", "PA", "H", "AVG", "1B", "2B", "3B", "HR", "R", "RBI", "BB", "SO" ]
) #Sidebar selectbox para seleccionar STAT a predecir

STAT_COL = selected_stat

batting = st.session_state["batting_ml"].copy() #Obtener valor desde sesión y evitar nuevamente la carga de datos
batting = batting.groupby("IDfg", group_keys=False).filter( lambda x: x.shape[0] > 2 ) #Solamente jugadores con más de dos temporadas registradas
st.dataframe( batting )

st.markdown(
    """
    Posteriormente se limpia el dataframe, se remueven columnas incompletas y se agrega una nueva columna,
    cuyo valor es la estadística seleccionada (del jugador) de la siguiente temporada, esto para cada uno de los registros.

    ```python
    #Limpia el dataframe de columnas incompletas y agrega una nueva columna para next_season
    batting = batting.groupby("IDfg", group_keys=False).apply( next_season )
    null_count = batting.isnull().sum()
    complete_cols = list( batting.columns[ null_count == 0 ] ) #Solamente columnas con todos los valores completos
    batting = batting[ complete_cols + [ f"Next_{ STAT_COL }" ]].copy()
    ```
    """
)

#Limpia el dataframe de columnas incompletas y agrega una nueva columna para next_season
batting = batting.groupby("IDfg", group_keys=False).apply( next_season )
null_count = batting.isnull().sum()
complete_cols = list( batting.columns[ null_count == 0 ] ) #Solamente columnas con todos los valores completos
batting = batting[ complete_cols + [ f"Next_{ STAT_COL }" ]].copy()

st.dataframe( batting )
#del batting["Age Rng"] #Columna no numérica innecesaria para el cálculo
#del batting["Dol"] #Columna no numérica innecesaria para el cálculo
#ML solo acepta numeros

st.markdown(
    """
    Agregamos team_code como valor numérico de Team para poder usarse en el modelo de Machine Learning y terminamos la limpieza
    eliminando las filas con registros nulos.

    ```python
    batting["team_code"] = batting["Team"].astype("category").cat.codes #Agrega una columna numérica basada en los distintos valores de Team (string)
    batting_full = batting.copy()
    batting = batting.dropna().copy()
    ```
    """
)

batting["team_code"] = batting["Team"].astype("category").cat.codes #Agrega una columna numérica basada en los distintos valores de Team (string)
batting = batting.dropna().copy()
batting_full = batting.copy()
st.dataframe( batting )

st.markdown(
    """
    Se define la configuración inicial para definir el proceso de modelado

    ```python
    rr = Ridge( alpha = 2 ) #Ridge regression estimation

    split = TimeSeriesSplit ( n_splits=3 )

    sfs = SequentialFeatureSelector(
        rr, 
        n_features_to_select = 20, 
        direction = "forward",
        cv = split,
        n_jobs = 8
    )
    ```

    """
)

rr = Ridge( alpha = 2 ) #Ridge regression estimation

split = TimeSeriesSplit ( n_splits=3 )

sfs = SequentialFeatureSelector(
    rr, 
    n_features_to_select = 20, 
    direction = "forward",
    cv = split,
    n_jobs = 8
)

removed_columns = [ f"Next_{ STAT_COL }", "Name", "Team", "Age Rng", "IDfg", "Dol", "Season", "Events", "L-WAR" ] #Columnas a no tomar en cuenta en la predicción de valor de STAT
selected_columns = batting.columns[ ~batting.columns.isin( removed_columns ) ]

st.markdown(
    """
    Se aplica un "scaler" para normalizar los datos entre 0 y 1, y aunque no es obligatorio, suele ser una buena práctica para
    posteriormente comparar resultados, incluso con otras implementaciones. Se termina aplicando un backtesting y se guardan
    los resultados en la lista "predictions"

    ```python
    scaler = MinMaxScaler() #Para escalar los valores de las columnas seleccionadas y se transformen con valores entre 0 y 1 (normalización)
    batting.loc[:,selected_columns] = scaler.fit_transform( batting[ selected_columns ] ) #Transformación con uso de scaler

    sfs.fit( batting[ selected_columns ], batting[ f"Next_{ STAT_COL }" ] ) #Training vector, Target value: queremos predecir STAT

    predictors = list( selected_columns[ sfs.get_support() ] ) #Crea una lista y asigna un index entero (int)
    predictions = backtest( batting, rr, predictors ) #backtesting
    ```
    """
)

scaler = MinMaxScaler() #Para escalar los valores de las columnas seleccionadas y se transformen con valores entre 0 y 1 (normalización)
batting.loc[:,selected_columns] = scaler.fit_transform( batting[ selected_columns ] ) #Transformación con uso de scaler

sfs.fit( batting[ selected_columns ], batting[ f"Next_{ STAT_COL }" ] ) #Training vector, Target value: queremos predecir STAT

predictors = list( selected_columns[ sfs.get_support() ] ) #Crea una lista y asigna un index entero (int) para los parametros seleccionados
predictions = backtest( batting, rr, predictors ) #backtesting

st.markdown(
    """
    La función backtesting se define como lo siguiente:
    
    ```python
    def backtest( data, model, predictors, start = 10, step = 1 ):
        '''
        Realiza un backtesting para predecir valores utilizando las temporadas anteriores como datos históricos
        Backtesting: es un término que se usa en modelado para referirse a probar un modelo predictivo utilizando datos históricos
        '''
        all_predictions = []
        years = sorted( data["Season"].unique() )

        #Entrena al modelo
        for i in range( start, len( years ), step ):

            current_year = years[ i ]
            train = data[ data["Season"] < current_year ] #train data = Años anteriores al año del registro seleccionado
            test = data[ data["Season"] == current_year ] #test data = Datos de año de registro seleccionado
            
            model.fit( train[ predictors ], train[ f"Next_{ STAT_COL }" ] ) #Training data, target values
            
            preds = model.predict( test[ predictors ] ) #Samples para predecir
            preds = pd.Series( preds, index=test.index )
            combined = pd.concat( [ test[ f"Next_{ STAT_COL }" ], preds ], axis = 1 ) #Agrega la columna de "predicción" al dataframe
            combined.columns = ["actual", "prediction"] #Solo para renombrar columnas
            
            all_predictions.append( combined ) #Guarda subset en lista

        return pd.concat( all_predictions ) #Concatena todos los subset en uno solo
    ```
    """
)

st.markdown(
    """
    Tras este backtesting estos son los resultados al aplicar la función mean_squared_error y r2_score de la librería sklearn:

    ```python
    error = mean_squared_error( predictions["actual"], predictions["prediction"] ) #Mean squared regression loss tras el primer backtesting
    r_squared = r2_score( predictions["actual"], predictions["prediction"] )
    ```
    """
)
error = mean_squared_error( predictions["actual"], predictions["prediction"] ) #Mean squared regression loss tras el primer backtesting
r_squared = r2_score( predictions["actual"], predictions["prediction"] )
st.write( "Mean squared error tras el primer backtesting; mean_squared_error = ", error )
st.write( "Accuracy tras el primer backtesting; r2_score = ", r_squared )

st.markdown(
    """
    Se agregan columnas calculadas para intentar mejorar la puntuación devuelta por r2_score.

    ```python
    def group_averages( df ):
        '''
        Comparación entre valor de STAT para el registro que accede y la media
        '''
        return df[ STAT_COL ] / df[ STAT_COL ].mean()

    def player_history(df):
        '''
        Agrega las columnas STAT_corr = Correlación y STAT_diff = diferencia entre valor y valor anterior (de la temporada pasada)
        '''
        df = df.sort_values("Season")
            
        df[ "player_season" ] = range( 0, df.shape[ 0 ] ) #Valores de 0 a N = Número de registros relacionados a este jugador (temporadas)

        expanded = df[ [ "player_season", STAT_COL ] ].expanding().corr() #Tabla de correlación expandida entre player_season y STAT
        reduced = expanded.loc[ ( slice( None ), "player_season" ), STAT_COL ] #Todos los registros de player_season con columnas player_season y STAT
        df[ f"{ STAT_COL }_corr" ] = list( reduced )

        #Calcula la correlación entre player_season y STAT
        df[ f"{ STAT_COL }_corr" ].fillna(0, inplace=True) #Rellena NaN con ceros
        
        df[ f"{ STAT_COL }_diff" ] = df[ STAT_COL ] / ( df[ STAT_COL ].shift(1) ) #Comparación entre valor de STAT entre fila y fila anterior
        df[ f"{ STAT_COL }_diff" ].fillna(1, inplace=True) 
        df[ f"{ STAT_COL }_diff" ][ df[ f"{ STAT_COL }_diff" ] == np.inf ] = 1 #Convertir resultado de división entre 0 a 1 (1 es valor máximo)
        
        return df


    batting = batting.groupby("IDfg", group_keys=False).apply( player_history )
    batting[ f"{ STAT_COL}_season" ] = batting.groupby("Season", group_keys=False).apply( group_averages )
    ```
    """
)

batting = batting.groupby("IDfg", group_keys=False).apply( player_history )
batting[ f"{ STAT_COL}_season" ] = batting.groupby("Season", group_keys=False).apply( group_averages )

st.markdown(
    """
    Se hace un nuevo backtesting con las nuevas columnas calculadas:

    ```python
    new_predictors = predictors + ["player_season", f"{ STAT_COL}_corr", f"{ STAT_COL}_season", f"{ STAT_COL}_diff"] #Agregar nuevas columnas calculadas al vector de predicciones
    predictions = backtest( batting, rr, new_predictors ) #backtesting
    ```

    con los siguientes resultados:

    """
)

new_predictors = predictors + ["player_season", f"{ STAT_COL}_corr", f"{ STAT_COL}_season", f"{ STAT_COL}_diff"] #Agregar nuevas columnas calculadas al vector de predicciones
predictions = backtest( batting, rr, new_predictors ) #backtesting

error = mean_squared_error( predictions["actual"], predictions["prediction"] ) #Mean squared regression loss tras el segundo backtesting
r_squared = r2_score( predictions["actual"], predictions["prediction"] )

st.write( "Mean squared error tras el segundo backtesting; mean_square_error = ", error )
st.write( "Accuracy tras el segundo backtesting; r2_score = ", r_squared )
st.markdown(
    """
    El resultado es variable según la estadística seleccionada, la siguiente configuración y dataframe es un compilado de los resultados
    para cada una de las estadísticas en el selectbox:

    ```python
    batting = batting_stats( 2002, 2023, qual=502 )
    rr = Ridge( alpha = 2 ) #Ridge regression estimation
    sfs = SequentialFeatureSelector(
        rr, 
        n_features_to_select = 20, 
        direction = "forward",
        cv = split,
        n_jobs = 8
    )
    def backtest( data, model, predictors, start = 10, step = 1 )

    error = mean_squared_error( predictions["actual"], predictions["prediction"] ) 
    r_squared = r2_score( predictions["actual"], predictions["prediction"] )
    ```
    """
)
results = pd.read_csv( Path( st.session_state['cur_path'] , 'tools/data', 'modeling_results.csv') , index_col=0 )
st.dataframe(  results )

pd.Series(rr.coef_, index=new_predictors).sort_values()
#diff = predictions["actual"] - predictions["prediction"]
merged = predictions.merge( batting, left_index=True, right_index=True )
#merged["prediction"] = predictions["prediction"]
merged["diff"] = ( predictions["actual"] - predictions["prediction"] ).abs()
merged_final = merged[["IDfg", "Season", "Name", STAT_COL, f"Next_{ STAT_COL}", "prediction", "diff"]].sort_values( [ "diff" ] ) #Ordenar por valor de diff de manera ascendente

st.markdown(
    """
    La predicción arroja una exactitud relativamente alta aplicada en la estadística de BB.
    El siguiente es el dataframe final tras hacer merging con los datos originales y el dataframe de predicción:
    """
)
st.dataframe( merged_final )
st.subheader("Conclusión")
st.markdown(
    """
    <div style='text-align: justify;'>
        Se concluye que el modelo no puede aplicarse por igual a todas las estadísticas, algunas de ellas retornan exactitudes bajas, mientras
        que unas cuantas son relativamente altas. Particularmente en béisbol, es en extremo complicado predecir un resultado por la inmensa cantidad
        de variables que afectan al juego, incluso factores externos como condiciones físicas, mentales y emocionales en los jugadores, se pueden
        reflejar en un cambio drástico en su rendimiento. Y, aunque el modelo aun se encuentra lejos de lo exacto, sí puede ser un factor
        determinante en la toma de decisiones. Estoy consciente que en la MLB se aplican modelos mucho más complejos que arrojan valores con
        mayor exactitud, aún así, este proyecto me permitió introducirme al mundo de la Ciencia de Datos y a muchas cosas más que desconocía en
        un principio; en las primeras clases del bootcamp, además me parece fascinante que todo ese conocimiento pueda aplicarse tan variadamente
        en un deporte que he seguido a detalle desde que tenía 7 años. Sin nada más que agregar, solo me queda agradecer al equipo de Código Facilito
        y a mis compañeros de generación por su apoyo y en la construcción de nuevo conocimiento.
        <br>
        <span style="font-weight: bold; font-size: 20px;">¡Gracias!</span>
    </div>
    """,
    unsafe_allow_html=True
)