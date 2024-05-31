from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from dash import html, dcc, Dash, Input, Output, State
import dash_bootstrap_components as dbc
# Initialize Flask app and Dash app

app = Flask(__name__)
dash_app = Dash(__name__, server=app, external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])

# Load model and preprocessing transformers
model = joblib.load("model.joblib")
fitted_column_transformer = joblib.load("column_transformer.joblib")
fitted_encoder = joblib.load("encoder.joblib")

# Define the names of numerical and categorical features
numerical_feature_names = ['Weight', 'Age', 'Height', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
categorical_feature_names = ['SMOKE', 'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'CALC', 'SCC', 'MTRANS']
cols = numerical_feature_names + categorical_feature_names

# Define layout for Dash app
dash_app.layout = dbc.Container([
    html.H1("Obesity Level Prediction", style={'textAlign': 'left', 'font-family': 'Helvetica', 'margin-bottom': '40px', 'margin-top': '40px'}),

    dbc.Row([
        dbc.Col([

            html.Div([
                dbc.Row(dbc.Col(html.H5("Please enter your info:", style={'textAlign': 'left', 'margin-bottom': '10px','color':'#8caca1'}), width=6)),
            ], style={'margin-bottom': '10px'}),
            html.Div([
                dbc.Row(dbc.Col(html.Label('Weight: '), width=6)),
                dbc.Row(dbc.Col(dcc.Input(id='Weight', type='number', value=70), width=6)),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                dbc.Row(dbc.Col(html.Label('Age: '), width=6)),
                dbc.Row(dbc.Col(dcc.Input(id='Age', type='number', value=30), width=6)),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                dbc.Row(dbc.Col(html.Label('Height: '), width=6)),
                dbc.Row(dbc.Col(dcc.Input(id='Height', type='number', value=180), width=6)),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                dbc.Row(dbc.Col(html.Label('Frequency of consuming vegetables: '), width=6)),
                dbc.Row(dbc.Col(dcc.Input(id='FCVC', type='number', value=0.1, step=0.1), width=6)),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                dbc.Row(dbc.Col(html.Label('Number of main meals consumed per day: '), width=6)),
                dbc.Row(dbc.Col(dcc.Input(id='NCP', type='number', value=3, step=1), width=6)),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                dbc.Row(dbc.Col(html.Label('Amount of water consumed daily: '), width=6)),
                dbc.Row(dbc.Col(dcc.Input(id='CH20', type='number', value=1.0, step=0.25), width=6)),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                dbc.Row(dbc.Col(html.Label('Frequency of engaging in physical activity: '), width=6)),
                dbc.Row(dbc.Col(dcc.Input(id='FAF', type='number', value=1.0, step=0.1), width=6)),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                dbc.Row(dbc.Col(html.Label('Time spent using technology device: '), width=6)),
                dbc.Row(dbc.Col(dcc.Input(id='TUE', type='number', value=1.0, step=0.1), width=6)),
            ], style={'margin-bottom': '10px'}),
        ], width=6),

        dbc.Col([
            html.Div([
                html.Label('Frequent consumption of high-caloric food items: '),
                dcc.Dropdown(options=[{'label': 'Yes', 'value': 'yes'}, {'label': 'No', 'value': 'no'}], id='FAVC', value='no'),
            ], style={'margin-top': '40px','margin-bottom': '10px'}),

            html.Div([
                html.Label('Frequency of alcohol consumption: '),
                dcc.Dropdown(options=[{'label': 'Frequently', 'value': 'Frequently'}, {'label': 'Sometimes', 'value': 'Sometimes'}, {'label': 'No', 'value': 'no'}], id='CALC', value='no'),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                html.Label('Frequency of consuming food between meals: '),
                dcc.Dropdown(options=[{'label': 'Always', 'value': 'Always'}, {'label': 'Frequently', 'value': 'Frequently'}, {'label': 'Sometimes', 'value': 'Sometimes'}, {'label': 'No', 'value': 'no'}], id='CAEC', value='no'),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                html.Label('Monitoring of calorie consumption: '),
                dcc.Dropdown(options=[{'label': 'Yes', 'value': 'yes'}, {'label': 'No', 'value': 'no'}], id='SCC', value='no'),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                html.Label('Transportation method used: '),
                dcc.Dropdown(options=[{'label': 'Automobile', 'value': 'Automobile'}, {'label': 'Bike', 'value': 'Bike'}, {'label': 'Motorbike', 'value': 'Motorbike'}, {'label': 'Public Transportation', 'value': 'Public_Transportation'}, {'label': 'Walking', 'value': 'Walking'}], id='MTRANS', value='Walking'),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                html.Label('Smoking status: '),
                dcc.Dropdown(options=[{'label': 'Yes', 'value': 'yes'}, {'label': 'No', 'value': 'no'}], id='SMOKE', value='no'),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                html.Label('Gender: '),
                dcc.Dropdown(options=[{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}], id='Gender', value='Male'),
            ], style={'margin-bottom': '10px'}),

            html.Div([
                html.Label('Family History with Overweight: '),
                dcc.Dropdown(options=[{'label': 'Yes', 'value': 'yes'}, {'label': 'No', 'value': 'no'}], id='family_history', value='no'),
            ], style={'margin-bottom': '10px'}),
    ])
        ]),
    dbc.Row(html.Div(id="output_prediction", style={'margin-top': '20px','textAlign': 'center'})),
    dbc.Row(dbc.Col([
        html.Div([
            dbc.Button(id="button", outline=True,color='info', size="lg", children="Submit")
        ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '10vh', 'margin-bottom': '30px'})
    ], width=12)),
    dbc.Row([
        dbc.Col([
            html.Div(id="alerts")
        ])
    ])
])


# Define callback function for Dash app
# Define callback function for Dash app
# Define callback function for Dash app
@dash_app.callback(
    Output(component_id="output_prediction", component_property="children"),
    [Input(component_id="button", component_property="n_clicks")],
    [State(component_id="Weight", component_property="value"),
     State(component_id="Age", component_property="value"),
     State(component_id="Height", component_property="value"),
     State(component_id="FCVC", component_property="value"),
     State(component_id="NCP", component_property="value"),
     State(component_id="CAEC", component_property="value"),
     State(component_id="CH20", component_property="value"),
     State(component_id="FAF", component_property="value"),
     State(component_id="TUE", component_property="value"),
     State(component_id="CALC", component_property="value"),
     State(component_id="FAVC", component_property="value"),
     State(component_id="SCC", component_property="value"),
     State(component_id="MTRANS", component_property="value"),
     State(component_id="SMOKE", component_property="value"),
     State(component_id="Gender", component_property="value"),
     State(component_id="family_history", component_property="value")]
)
def predict(n_clicks, weight, age, height, fcvc, ncp, caec, ch20, faf, tue, calc, favc, scc, mtrans, smoke, gender, family_history):
    if n_clicks is None:
        return None
    else:
        # Prepare data for prediction
        feature_values = {
            'Weight': float(weight),
            'Age': float(age),
            'Height': float(height),
            'FCVC': float(fcvc),
            'NCP': float(ncp),
            'CH2O': float(ch20),
            'FAF': float(faf),
            'TUE': float(tue),
            'CALC': calc,
            'FAVC': favc,
            'CAEC': caec,
            'SCC': scc,
            'MTRANS': mtrans,
            'SMOKE': smoke,
            'Gender': gender,
            'family_history_with_overweight': family_history
        }

        # Stack numerical and categorical features horizontally
        sample = pd.DataFrame([feature_values], columns=cols)
        transformed_features = fitted_column_transformer.transform(sample)

        # Make prediction
        prediction = model.predict(transformed_features)

        # Transform predicted target values back to original categories
        transformed_prediction = fitted_encoder.inverse_transform(prediction.reshape(-1, 1)).flatten()
        predicted_obesity_level = transformed_prediction[0]
        # Generate alert message based on predicted obesity level
        if predicted_obesity_level == 'Normal_Weight':
            alert_color = "success"
            alert_message = f" Predicted Obesity Level: {predicted_obesity_level.replace('_', ' ')}"
            alert_icon = "bi bi-check-circle-fill"

        elif predicted_obesity_level == 'Insufficient_Weight':
            alert_color = "warning"
            alert_message = f" Predicted Obesity Level: {predicted_obesity_level.replace('_', ' ')}"
            alert_icon = "bi bi-exclamation-triangle-fill me-2"

        elif predicted_obesity_level in ['Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']:
            alert_color = "danger"
            alert_message = f" Predicted Obesity Level: {predicted_obesity_level.replace('_', ' ')}"
            alert_icon = "bi bi-x-octagon-fill me-2"
        else:
            alert_color = "info"
            alert_message = "Invalid prediction"
            alert_icon = "bi bi-x-circle-fill"

        # Create the alert component
        alert = dbc.Alert(
            [
                html.I(className=alert_icon),
                alert_message
            ],
            color=alert_color,
            className="d-flex align-items-center"
        )

        # Return prediction result with integrated alert
        return html.Div([alert])




if __name__ == '__main__':
    app.run(debug=True, port=8053)