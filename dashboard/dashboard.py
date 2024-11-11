import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('processed_news_dataset.csv')

# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
app.layout = html.Div(children=[
    html.H1(children='Dashboard de Análisis de Sentimientos', style={'text-align': 'center'}),

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
