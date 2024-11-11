import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import mlflow
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Cargar los experimentos desde MLflow
experiment_name = 'Sentiment_Analysis'  # Nombre de tu experimento en MLflow
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
runs = mlflow.search_runs(experiment_ids=[experiment_id])

# Obtener las métricas registradas de MLflow
coef_1 = runs['coef_1'].iloc[0] if 'coef_1' in runs.columns else 'No data'
coef_2 = runs['coef_2'].iloc[0] if 'coef_2' in runs.columns else 'No data'

# Cargar el archivo CSV para crear las visualizaciones adicionales
df = pd.read_csv('processed_news_dataset.csv')

# Crear visualizaciones

# Gráfico de barras para contar la cantidad de diferentes tipos de palabras por categoría
fig1 = px.bar(df, x='category', y=['adj_count', 'verb_count', 'noun_count', 'proper_noun_count', 'adv_count'],
              barmode='group', title="Distribución de Tipos de Palabras por Categoría")

# Gráfico de líneas para mostrar la evolución de la cantidad de adjetivos a lo largo del tiempo
df['date'] = pd.to_datetime(df['date'])
fig2 = px.line(df, x='date', y='adj_count', title="Evolución de la Cantidad de Adjetivos por Fecha")

# Gráfico de dispersión para explorar la relación entre el número de adjetivos y sustantivos
fig3 = px.scatter(df, x='adj_count', y='noun_count', color='category', title="Relación entre Adjetivos y Sustantivos")

# Crear la nube de palabras a partir de 'processed_text'
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['processed_text'].dropna()))
fig4 = plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud.png')  # Guardar la imagen de la nube de palabras

# **Gráfico adicional: Categorías por Sentimiento**
# Contar las ocurrencias de cada sentimiento por categoría
sentiment_count = df.groupby(['category', 'sentiment_text']).size().reset_index(name='count')

# Crear un gráfico de barras para mostrar la cantidad de categorías por sentimiento
fig5 = px.bar(sentiment_count, x='category', y='count', color='sentiment_text',
              barmode='stack', title="Categorías por Sentimiento")

# Crear el layout del dashboard
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    html.H1(children='Dashboard de Análisis de Sentimientos con MLflow', style={'text-align': 'center'}),

    # Mostrar las métricas registradas desde MLflow
    dbc.Row([
        dbc.Col(html.H3(f'Coeficiente 1: {coef_1}'), width=4),
        dbc.Col(html.H3(f'Coeficiente 2: {coef_2}'), width=4),
    ], style={'margin-bottom': '40px'}),

    # Fila 1: Gráfico de barras (tipos de palabras por categoría)
    dbc.Row([
        dbc.Col(dcc.Graph(id='bar-graph', figure=fig1), width=12)
    ], style={'margin-bottom': '40px'}),

    # Fila 2: Gráfico de líneas (evolución de adjetivos por fecha)
    dbc.Row([
        dbc.Col(dcc.Graph(id='line-graph', figure=fig2), width=12)
    ], style={'margin-bottom': '40px'}),

    # Fila 3: Gráfico de dispersión (relación entre adjetivos y sustantivos)
    dbc.Row([
        dbc.Col(dcc.Graph(id='scatter-graph', figure=fig3), width=12)
    ], style={'margin-bottom': '40px'}),

    # Fila 4: Nube de palabras
    dbc.Row([
        dbc.Col(html.Img(src='/assets/wordcloud.png', style={'width': '100%'}), width=12)
    ], style={'margin-bottom': '40px'}),

    # Fila 5: Gráfico de categorías por sentimiento
    dbc.Row([
        dbc.Col(dcc.Graph(id='sentiment-graph', figure=fig5), width=12)
    ], style={'margin-bottom': '40px'}),
])

# Ejecutar el servidor
if __name__ == '__main__':
    app.run_server(debug=True)
