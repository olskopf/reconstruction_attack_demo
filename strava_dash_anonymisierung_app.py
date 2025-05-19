import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Verfügbare Datensatzversionen
dataset_paths = {
    'No Anonymization': 'data/strava_with_demographics.csv',
    'k=2': 'data/strava_anonymized_k2.csv',
    'k=5': 'data/strava_anonymized_k5.csv',
    'k=20': 'data/strava_anonymized_k20.csv',
    'k=50': 'data/strava_anonymized_k50.csv'
}

# Zielvariablen (QIDs)
target_columns = ['age', 'height_cm', 'weight_kg']

# Eingabemerkmale
input_features = ['average_speed', 'max_speed', 'distance', 'total_elevation_gain', 'elev_high', 'kudos_count']
user_input_features = ['average_speed', 'max_speed', 'distance']

# App initialisieren
app = dash.Dash(__name__)
app.title = "Rekonstruktionsangriff Demo Strava"
dataset_models = {}

# Layout
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='train', children=[
        dcc.Tab(label='Modelltraining', value='train'),
        dcc.Tab(label='Datenansicht', value='view'),
        dcc.Tab(label='Eigene Eingabe', value='predict')
    ]),
    html.Div(id='tab-content')
])

# Tab-Inhalte
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def display_tab(tab):
    if tab == 'train':
        return html.Div([
            html.Label("Anteil der Trainingsdaten (%):"),
            dcc.Slider(id='train-percent', min=10, max=90, step=10, value=20,
                       marks={i: f'{i}%' for i in range(10, 100, 10)}),
            html.Label("Anzahl Bäume im Random Forest:"),
            dcc.Slider(id='n-trees', min=10, max=500, step=10, value=100,
                       marks={i: str(i) for i in range(10, 501, 50)}),
            html.Button('Trainieren', id='train-btn', n_clicks=0),
            html.Div(id='train-status'),
            html.Hr(),
            html.Label("Datensatz für Angriff auswählen:"),
            dcc.Dropdown(id='attack-set',
                         options=[{'label': k, 'value': k} for k in dataset_paths],
                         value='k=2'),
            html.Button('Angriff starten', id='attack-btn', n_clicks=0),
            html.Div(id='attack-status'),
            dash_table.DataTable(id='attack-metrics', style_table={'overflowX': 'auto'}),
            dcc.Graph(id='plot-age'),
            dcc.Graph(id='plot-weight'),
            dcc.Graph(id='plot-height'),
        ])
    elif tab == 'predict':
        return html.Div([
            html.H3("Eigene Eingabe testen (Einheiten beachten):"),
            html.Label("Durchschnittsgeschwindigkeit [m/s]"),
            dcc.Input(id='input-avg', type='number', value=3.0, step=0.1),
            html.Label("Maximale Geschwindigkeit [m/s]"),
            dcc.Input(id='input-max', type='number', value=5.0, step=0.1),
            html.Label("Distanz [m]"),
            dcc.Input(id='input-dist', type='number', value=8000, step=100),
            html.Div(id='predict-output')
        ])
    else:
        return html.Div([
            html.Label("Datensatz anzeigen:"),
            dcc.Dropdown(id='view-set',
                         options=[{'label': k, 'value': k} for k in dataset_paths],
                         value='No Anonymization'),
            dash_table.DataTable(
                id='data-table',
                page_size=15,
                style_table={'overflowX': 'auto'},
                style_data_conditional=[]  # wird im Callback gesetzt
            )
        ])

# Intervallparser
def parse_interval(val):
    if isinstance(val, str) and '-' in val:
        parts = val.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

# Metrikberechnung
def create_metrics(y_true, y_pred, name):
    return {
        'QID': name,
        'MAE': round(np.mean(np.abs(y_true - y_pred)), 2),
        '≤2%': f"{np.mean(np.abs(y_true - y_pred) <= y_true * 0.02):.0%}",
        '≤5%': f"{np.mean(np.abs(y_true - y_pred) <= y_true * 0.05):.0%}",
        '≤10%': f"{np.mean(np.abs(y_true - y_pred) <= y_true * 0.10):.0%}",
        '≤20%': f"{np.mean(np.abs(y_true - y_pred) <= y_true * 0.20):.0%}"
    }

# Modelltraining
@app.callback(Output('train-status', 'children'),
              Input('train-btn', 'n_clicks'),
              Input('train-percent', 'value'),
              Input('n-trees', 'value'))
def train_model(n_clicks, percent, n_trees):
    if n_clicks == 0:
        return ""
    df = pd.read_csv(dataset_paths['No Anonymization']).dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[input_features])

    models = {}
    for target in target_columns:
        y = df[target].values
        X_train, _, y_train, _ = train_test_split(X, y, train_size=percent/100, random_state=1)
        model = RandomForestRegressor(n_estimators=n_trees, random_state=1)
        model.fit(X_train, y_train)
        models[target] = model

    global dataset_models
    models['scaler'] = scaler
    dataset_models = models
    return f"Training abgeschlossen mit {n_trees} Bäumen"

# Angriff
@app.callback([
    Output('attack-status', 'children'),
    Output('attack-metrics', 'data'),
    Output('attack-metrics', 'columns'),
    Output('plot-age', 'figure'),
    Output('plot-weight', 'figure'),
    Output('plot-height', 'figure')
],
    Input('attack-btn', 'n_clicks'),
    Input('attack-set', 'value'))
def attack_model(n_clicks, dataset_key):
    if n_clicks == 0 or not dataset_models:
        return "", [], [], go.Figure(), go.Figure(), go.Figure()

    df_anon = pd.read_csv(dataset_paths[dataset_key], index_col=0).dropna()
    df_real = pd.read_csv(dataset_paths['No Anonymization'], index_col=0).dropna()

    for col in target_columns:
        if col in df_anon.columns:
            df_anon[col] = df_anon[col].apply(parse_interval)

    X_scaled = dataset_models['scaler'].transform(df_anon[input_features])
    df_real = df_real.loc[df_anon.index]

    table, plots = [], {}
    for col in target_columns:
        model = dataset_models[col]
        y_true = df_real[col].values
        y_pred = model.predict(X_scaled).flatten()
        table.append(create_metrics(y_true, y_pred, col))

        fig = go.Figure([
            go.Scatter(y=y_true, mode='markers', name='Echt', marker=dict(color='blue')),
            go.Scatter(y=y_pred, mode='markers', name='Vorhersage', marker=dict(color='red'))
        ])
        fig.update_layout(title=f'{col}: Vorhersage vs Echte Werte', xaxis_title='Index', yaxis_title=col)
        plots[col] = fig

    columns = [{'name': i, 'id': i} for i in table[0].keys()]
    return "Angriff abgeschlossen", table, columns, plots['age'], plots['weight_kg'], plots['height_cm']

# Eigene Eingabe
@app.callback(Output('predict-output', 'children'),
              Input('input-avg', 'value'),
              Input('input-max', 'value'),
              Input('input-dist', 'value'))
def predict_values(avg, maxv, dist):
    if not dataset_models:
        return ""
    scaler = dataset_models['scaler']
    df = pd.DataFrame([{
        'average_speed': avg,
        'max_speed': maxv,
        'distance': dist,
        'total_elevation_gain': 0,
        'elev_high': 0,
        'kudos_count': 0
    }])
    X_scaled = scaler.transform(df[input_features])

    predictions = {col: round(dataset_models[col].predict(X_scaled)[0], 2) for col in target_columns}
    return html.Div([
        html.P(f"Alter: {predictions['age']} Jahre"),
        html.P(f"Größe: {predictions['height_cm']} cm"),
        html.P(f"Gewicht: {predictions['weight_kg']} kg")
    ])

# Datensatz anzeigen – mit Stil
@app.callback(
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Output('data-table', 'style_data_conditional'),
    Input('view-set', 'value')
)
def show_table(dataset_key):
    df = pd.read_csv(dataset_paths[dataset_key])
    columns = [{'name': i, 'id': i} for i in df.columns]

    highlight_columns = ['age', 'gender', 'height_cm', 'weight_kg']
    styles = [
        {
            'if': {'column_id': col},
            'backgroundColor': '#ffeaa7',
            'color': 'black',
            'fontWeight': 'bold'
        } for col in highlight_columns
    ]

    return df.to_dict('records'), columns, styles

# Start
if __name__ == '__main__':
    app.run(debug=True)
