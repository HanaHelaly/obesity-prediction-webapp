from flask import Flask, request, jsonify
import joblib
import pandas as pd
from dash import html, dcc, Dash, Input, Output, State
import dash_bootstrap_components as dbc

# Initialize Flask app and Dash app
app = Flask(__name__)
dash_app = Dash(__name__, server=app, external_stylesheets=[dbc.themes.SLATE])

# Load model and preprocessing transformers
model = joblib.load("model.joblib")
fitted_column_transformer = joblib.load("column_transformer.joblib")
fitted_encoder = joblib.load("encoder.joblib")

# Define the names of numerical and categorical features
numerical_feature_names = ['Weight', 'Age', 'Height', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
categorical_feature_names = ['SMOKE', 'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'CALC', 'SCC', 'MTRANS']
cols = numerical_feature_names + categorical_feature_names

# Define layout for Dash app
dash_app.layout = html.Div([
        dbc.Container([
            dbc.Row([
            html.H1("Obesity Level Prediction", style={'textAlign': 'center','color':'white','margin-top': '28px', 'margin-bottom': '50px'}),
            html.H2("Please enter your info:", style={ 'margin-bottom': '20px','font-size':20,'margin-top': '30px','margin-left': '100px'})
            ], style={'margin-top': '20px'}),
        dbc.Row([
            html.Div(style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}, children=[
                html.Div([
                    html.Label('Weight: '),
                    dcc.Input(id='Weight', type='number', value=50),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Age: '),
                    dcc.Input(id='Age', type='number', value=30),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Height: '),
                    dcc.Input(id='Height', type='number', value=150),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Frequency of consuming vegetables: '),
                    dcc.Input(id='FCVC', type='number', value=0.1, step=0.1),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Number of main meals consumed per day: '),
                    dcc.Input(id='NCP', type='number', value=3, step=1),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Amount of water consumed daily: '),
                    dcc.Input(id='CH20', type='number', value=0.25, step=0.25),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Frequency of engaging in physical activity: '),
                    dcc.Input(id='FAF', type='number', value=0.1, step=0.1),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Time spent using technology device: '),
                    dcc.Input(id='TUE', type='number', value=0.1, step=0.1),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Frequent consumption of high-caloric food items: '),
                    dcc.Dropdown(options=[{'label': 'Yes', 'value': 'yes'}, {'label': 'No', 'value': 'no'}], id='FAVC', value='no'),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Frequency of alcohol consumption: '),
                    dcc.Dropdown(options=[{'label':'Frequently','value':'Frequently'},{'label':'Sometimes','value':'Sometimes'},{'label': 'No', 'value': 'no'}], id='CALC', value='no'),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Frequency of consuming food between meals: '),
                    dcc.Dropdown(options=[{'label': 'Always', 'value': 'Always'},{'label':'Frequently','value':'Frequently'},{'label':'Sometimes','value':'Sometimes'},{'label': 'No', 'value': 'no'}], id='CAEC', value='no'),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Monitoring of calorie consumption: '),
                    dcc.Dropdown(options=[{'label': 'Yes', 'value': 'yes'}, {'label': 'No', 'value': 'no'}], id='SCC', value='no'),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Transportation method used: '),
                    dcc.Dropdown(options=[{'label': 'Automobile', 'value': 'Automobile'}, {'label': 'Bike', 'value': 'Bike'}, {'label': 'Motorbike', 'value': 'Motorbike'}, {'label': 'Public Transportation', 'value': 'Public_Transportation'}, {'label': 'Walking', 'value': 'Walking'}], id='MTRANS', value='Walking'),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Smoking status: '),
                    dcc.Dropdown(options=[{'label': 'Yes', 'value': 'yes'}, {'label': 'No', 'value': 'no'}], id='SMOKE', value='no'),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Gender: '),
                    dcc.Dropdown(options=[{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}], id='Gender', value='Male'),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    html.Label('Family History with Overweight: '),
                    dcc.Dropdown(options=[{'label': 'Yes', 'value': 'yes'}, {'label': 'No', 'value': 'no'}], id='family_history', value='no'),
                ], style={'margin-bottom': '10px', 'width': '50%'}),

                html.Div([
                    dbc.Button(id="button", color="info", children="Submit"),
                ], style={'margin-top': '20px'}),
            ]),

            html.Div(id="output_prediction", style={'margin-top': '20px', 'textAlign': 'center'})
        ], style={'margin-top': '20px'})
        ])
], style={'margin-bottom': '20px'})

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
def predict(n_clicks, weight, age, height, fcvc, ncp, caec, ch20, faf, tue, calc,favc, scc, mtrans, smoke, gender, family_history):
    if n_clicks is None:
        return "Waiting for input..."
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
            'FAVC':favc,
            'CAEC': caec,
            'SCC': scc,
            'MTRANS': mtrans,
            'SMOKE': smoke,
            'Gender': gender,
            'family_history_with_overweight': family_history
        }

        # Stack numerical and categorical features horizontally
        sample = pd.DataFrame([feature_values], columns=cols)
        print(sample.iloc[:,8:])
        transformed_features = fitted_column_transformer.transform(sample)
        print(transformed_features)

        # Make prediction
        prediction = model.predict(transformed_features)
        # Transform predicted target values back to original categories
        transformed_prediction = fitted_encoder.inverse_transform(prediction.reshape(-1, 1)).flatten()
        prediction_json = transformed_prediction.tolist()
        # Return prediction result as JSON response
        label = ' '.join((prediction_json[0]).split('_'))
        return f"Predicted Obesity Level: {label}"

# Run Flask server
if __name__ == '__main__':
    app.run(debug=True)
