# %%
# Standard library imports
import numpy as np
import pandas as pd

# Third-party library imports
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# %%
df = pd.read_csv('heart_2020_cleaned_model_2.csv')
df

# %%
df.info(
)

# %%
df1 = df.copy()

# %%
features_to_scale_robust = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
robust_scaler = RobustScaler()
df1[features_to_scale_robust] = robust_scaler.fit_transform(df1[features_to_scale_robust])

# %%
X = df1.drop(['HeartDisease'], axis=1)
y = df1['HeartDisease']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scaler =MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy_score(y_test, y_pred)*100


# %%
# Load the dataset
df = pd.read_csv('heart_2020_cleaned_.csv')

# Define diseases columns
diseases_columns = ['Stroke', 'DiffWalking', 'Diabetic', 'Asthma', 'KidneyDisease', 'SkinCancer']

# Initialize an empty dictionary to store contingency tables for each disease
contingency_tables = {}

# Create a contingency table (cross-tabulation) for each disease
for disease in diseases_columns:
    contingency_tables[disease] = pd.crosstab(index=df[disease], columns=df['HeartDisease'])

# Filter data for individuals with heart disease
heart_disease_df = df[df['HeartDisease'] == 'Yes']

# Calculate the count of individuals by race
race_counts = heart_disease_df['Race'].value_counts().sort_values(ascending=False)

# Define colorscale with blue shades
colorscale = [
    [0.0, 'rgb(204, 229, 255)'],  # Light blue
    [0.5, 'rgb(0, 102, 204)'],    # Medium blue
    [1.0, 'rgb(0, 51, 102)']      # Dark blue
]

# Create the bar trace
bar_trace = go.Bar(
    x=race_counts.index,
    y=race_counts.values,
    marker=dict(color=race_counts.values, colorscale=colorscale),
)

# Define layout for the Dash app
layout1 = go.Layout(
    title='Heart Disease Counts by Race (Descending Order)',
    xaxis=dict(title='Race'),
    yaxis=dict(title='Count'),
    plot_bgcolor='rgba(0,0,0,0)'
)

# Create the figure
fig1 = go.Figure(data=[bar_trace], layout=layout1)

# Calculate the count of individuals by race and sex
race_sex_counts = heart_disease_df.groupby(['Race', 'Sex']).size().unstack(fill_value=0)

# Calculate the total count of individuals by race
race_totals = race_sex_counts.sum(axis=1)

# Calculate the percentage of individuals by sex within each race
race_sex_percentages = race_sex_counts.div(race_totals, axis=0) * 100

# Define colors for males and females
color_male = 'cornflowerblue'
color_female = 'lightblue'

# Create the bar traces for males and females
bar_trace_male = go.Bar(
    x=race_sex_percentages.index,
    y=race_sex_percentages['Male'],
    name='Male',
    marker=dict(color=color_male),
    text=race_sex_percentages['Male'].round(2).astype(str) + '%',
    hoverinfo='text'
)

bar_trace_female = go.Bar(
    x=race_sex_percentages.index,
    y=race_sex_percentages['Female'],
    name='Female',
    marker=dict(color=color_female),
    text=race_sex_percentages['Female'].round(2).astype(str) + '%',
    hoverinfo='text'
)

# Define layout for the Dash app
layout2 = go.Layout(
    title='Heart Disease Percentage Distribution by Race and Gender',
    xaxis=dict(title='Race'),
    yaxis=dict(title='Percentage'),
    barmode='stack',  # Stack bars on top of each other
    plot_bgcolor='rgba(0,0,0,0)'
)

# Create the figure
fig2 = go.Figure(data=[bar_trace_male, bar_trace_female], layout=layout2)

# %%
# Define the race mapping dictionary
race_mapping = {
    'American Indian/Alaskan Native': 0,
    'Asian': 1,
    'Black': 2,
    'Hispanic': 3,
    'Other': 4,
    'White': 5
}

# Define the age category mapping dictionary
age_mapping = {
    '18-24': 0,
    '25-29': 1,
    '30-34': 2,
    '35-39': 3,
    '40-44': 4,
    '45-49': 5,
    '50-54': 6,
    '55-59': 7,
    '60-64': 8,
    '65-69': 9,
    '70-74': 10,
    '75-79': 11,
    '80 or older': 12
}

# Define the general health mapping dictionary
gen_health_mapping = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very good': 3,
    'Excellent': 4
}

# Define the physical activity mapping dictionary
physical_activity_mapping = {
    'No': 0,
    'Yes': 1
}

# Initialize the Dash app
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
server = app.server()

# Define custom CSS styles for dropdowns and input fields
dropdown_style = {
    'width': '100%',  # Set the width of the dropdown to 100% of the container
    'verticalAlign': 'left'  # Align the dropdown vertically in the middle
}

input_style = {
    'width': '100%',  # Set the width of the input field to 100% of the container
    'height': '35px',  # Set the height of the input field
    'verticalAlign': 'middle',  # Align the input field vertically in the middle
    'display': 'block'
}

# Define custom CSS styles for the main content area
content_style = {
    'width': '66.67%',  # Set the width of the content area to 2/3 of the container (100/3 * 2)
    'display': 'inline-block',
    'verticalAlign': 'top',
    'padding': '20px',
    # 'borderLeft': '1px solid #ccc',  # Add a solid border on the left side
    
}

# Define custom CSS styles for the sidebar
sidebar_style = {
    'width': '33.33%',  # Set the width of the sidebar to 1/3 of the container (100/3 * 1)
    'display': 'inline-block',
    'verticalAlign': 'top',
    'padding': '20px',
    'borderRight': '1px solid #ccc',
     'backgroundColor': 'lightblue'
}
sidebar_label_style = {
    'fontSize': '20px',  # Set the font size to 18 pixels
    'fontWeight': 'bold',  # Make the font bold
    'display':'block'
}


# Layout of the app
app.layout = html.Div([
    html.H1("Heart Disease Prediction",  style={'textAlign': 'center', 'color': 'black', 'background': 'lightblue', 'padding': '20px'}),
    
    dcc.Tabs([
        dcc.Tab(label='Model', children=[
            # Put your model tab content here
            html.Div([
    
    
    # Sidebar with inputs
    html.Div(style=sidebar_style, children=[
        html.H2("Input Features"),
          # Race dropdown
        html.Div([
            html.Label("Race", style=sidebar_label_style),
            dcc.Dropdown(
                id='input-race',
                options=[{'label': race, 'value': value} for race, value in race_mapping.items()],
                placeholder='Select Race',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between Race dropdown and Sex dropdown
        ]),
        
        # Sex dropdown
        html.Div([
            html.Label("Sex", style=sidebar_label_style),
            dcc.Dropdown(
                id='input-sex',
                options=[
                    {'label': 'Female', 'value': 0},
                    {'label': 'Male', 'value': 1}
                ],
                placeholder='Select Sex',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between Sex dropdown and AgeCategory dropdown
        ]),

         # AgeCategory dropdown
        html.Div([
            html.Label("Age Category", style=sidebar_label_style),
            dcc.Dropdown(
                id='input-age-category',
                options=[{'label': age_cat, 'value': value} for age_cat, value in age_mapping.items()],
                placeholder='Select Age Category',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between AgeCategory dropdown and BMI input
        ]),
        
        # BMI input
        html.Div([
            html.Label("BMI", style=sidebar_label_style),
            dcc.Input(id='input-bmi', type='number', placeholder='Enter your BMI', step=0.01, style=input_style),
            html.Br()  # Add an empty line between BMI input and MentalHealth input
        ]),

         # MentalHealth input
        html.Div([
            html.Label(
            "For how many days during the past 30 days",
            style=sidebar_label_style
            ),
            html.Span(
                "was your mental health not good?",
                style=sidebar_label_style
            ),
            dcc.Input(id='input-mental-health', type='number', placeholder='Enter Mental Health', step=0.01, style=input_style),
            html.Br()  # Add an empty line between MentalHealth input and PhysicalHealth input
        ]),

        # PhysicalHealth input
        html.Div([
            html.Label(
            "For how many days during the past 30 days",
            style=sidebar_label_style
            ),
            html.Span(
                "was your physical health not good?",
                style=sidebar_label_style
            ),
            dcc.Input(id='input-physical-health', type='number', placeholder='Enter Physical Health', step=0.01, style=input_style),
            html.Br()  # Add an empty line between PhysicalHealth input and PhysicalActivity dropdown
        ]),

          # SleepTime input
        html.Div([
            html.Label("How many hours on average do you sleep?", style=sidebar_label_style),
            dcc.Input(id='input-sleep-time', type='number', placeholder='Enter Sleep Time (hours)', style=input_style),
            html.Br()
        ]),

        # PhysicalActivity dropdown (Yes/No)
        html.Div([
             html.Label(
            "Have you played any sports like running,",
            style=sidebar_label_style
            ),
            html.Span(
                "biking in the past month?",
                style=sidebar_label_style
            ),
            dcc.Dropdown(
                id='input-physical-activity',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                placeholder='Select Physical Activity Status',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between PhysicalActivity dropdown and Diabetic dropdown
        ]),

        # Diabetic dropdown (Yes/No)
        html.Div([
            html.Label("Have you ever had diabetes?", style=sidebar_label_style),
            dcc.Dropdown(
                id='input-diabetic',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                placeholder='Select Diabetic Status',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between Diabetic dropdown and Asthma dropdown
        ]),

        # Asthma dropdown (Yes/No)
        html.Div([
            html.Label("Do you have asthma?", style=sidebar_label_style),
            dcc.Dropdown(
                id='input-asthma',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                placeholder='Select Asthma Status',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between Asthma dropdown and KidneyDisease dropdown
        ]),

            # KidneyDisease dropdown (Yes/No)
        html.Div([
            html.Label("Do you have kidney disease?", style=sidebar_label_style),
            dcc.Dropdown(
                id='input-kidney-disease',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                placeholder='Select Kidney Disease Status',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between KidneyDisease dropdown and Smoking dropdown
        ]),
        
        # Smoking dropdown
        html.Div([
            html.Label("Have you smoked before in your entire life?", style=sidebar_label_style),
            dcc.Dropdown(
                id='input-smoking',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                placeholder='Select Smoking Status',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between Smoking dropdown and Alcohol Drinking dropdown
        ]),

       # Alcohol Drinking dropdown
        html.Div([
             html.Label(
            "Do you have more than 14 drinks of alcohol",
            style=sidebar_label_style
            ),
            html.Span(
                "(men)or more than 7 (women) in a week?",
                style=sidebar_label_style
            ),
           
            dcc.Dropdown(
                id='input-alcohol',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                placeholder='Select Alcohol Drinking Status',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between Alcohol Drinking dropdown and Stroke dropdown
        ]),

         # Stroke dropdown
        html.Div([
            html.Label("Did you have a stroke?", style=sidebar_label_style),
            dcc.Dropdown(
                id='input-stroke',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                placeholder='Select Stroke Status',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between Stroke dropdown and Submit button
        ]),

        # Skin Cancer dropdown
        html.Div([
            html.Label("Do you have skin cancer?", style=sidebar_label_style),
            dcc.Dropdown(
                id='input-skin-cancer',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                placeholder='Select Skin Cancer Status',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between Skin Cancer dropdown and Submit button
        ]),

          # DiffWalking dropdown (Yes/No)
        html.Div([
             html.Label(
            "Do you have serious difficulty walking or ",
            style=sidebar_label_style
            ),
            html.Span(
                " climbing stairs?",
                style=sidebar_label_style
            ),
            dcc.Dropdown(
                id='input-diff-walking',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                placeholder='Select Difficulty Walking',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between DiffWalking dropdown and Smoking dropdown
        ]),
        
        # GeneralHealth dropdown
        html.Div([
            html.Label("How can you define your general health?", style=sidebar_label_style),
            dcc.Dropdown(
                id='input-general-health',
                options=[{'label': health, 'value': value} for health, value in gen_health_mapping.items()],
                placeholder='Select General Health Status',
                style=dropdown_style
            ),
            html.Br()  # Add an empty line between GeneralHealth dropdown and Diabetic dropdown
        ]),
        
        
    ]),
    
    # Main content area for submit button and result
    html.Div(style=content_style, children=[
        html.Div([
            html.H4(style={
        'font-weight': 'bold',  # Make the text bold
        'font-size': '24px',  # Set the font size to 24 pixels
        'color': 'black',  # Set the text color to blue
        'text-align': 'center',  # Align the text to the center
        'margin-top': '20px',  # Add top margin of 20 pixels
        'margin-bottom': '10px',  # Add bottom margin of 10 pixel
        'font-size': '30px',
        'margin': '10px 60px'
        },
      children=[
            "Are you wondering about the condition of your heart? ",
            "This app will help you diagnose it!"
            ]),
            
            html.Img(src='https://t4.ftcdn.net/jpg/06/14/96/05/360_F_614960515_mQsF7nS1r3qZ9eCHzqJ5cyCxmjsfJOCQ.jpg', 
                style={
                    'width': '40%',  # Adjust width to 40% of container
                    'height': 'auto',  # Maintain aspect ratio
                    'maxWidth': '400px',  # Limit width to 400 pixels
                    'marginBottom': '10px',  # Add bottom margin for spacing
                    'marginLeft': '360px',  # Center align image horizontally
                    'marginRight': 'auto'  # Center align image horizontally

                }
            ),
            
            html.Div([
                html.P("Welcome to our Heart Disease Risk Prediction App!",style={'text-align': 'left','font-size': '25px','margin': '10px 60px'}),
                html.P("Did you know that machine learning models can accurately assess your risk of heart disease? With this app, you can estimate your likelihood of developing heart disease (High/Low) in just a few seconds!",style={'font-size': '22px','margin': '10px 60px'}),
                html.P("To predict your heart disease status, simply follow the steps bellow:",style={'font-size': '22px','margin': '10px 60px'}),
                html.Ol([ 
                    html.Li("Enter the parameters that best describe you."),
                    html.Li("Press the Predict button and wait for the result."),    
                ], style={'text-align': 'left','font-size': '22px','margin': '10px 60px'}),
                html.P("Please remember that this prediction is not a substitute for a medical diagnosis. Our model is not intended for use in healthcare facilities due to its less than perfect accuracy. If you have any health concerns, please consult a qualified medical professional.",
                        style = {'margin': '10px 60px','font-size': '25px','font-weight': 'bold','text-align': 'left','padding': '10px',  'border': '2px solid #007BFF', 'border-radius': '5px','background-color': '#E3F2FD'  })

            ]),
            html.Button('Predict', id='submit-val', n_clicks=0, className='btn btn-primary btn-lg', style={'margin': '10px 60px','fontSize': '30px', 'padding': '16px', 'width': '200px', 'height': '100px'}),
            html.Div(id='output-container', style={'margin-top': '20px','margin': '10px 60px'})
        ])
    ])
])

        ]),

        dcc.Tab(label='Visualizations', children=[
            # Content for the visualizations tab
            html.H2("General Health", style={'textAlign': 'center'}),
                  
            html.Div([
                html.H3("Physical and Mental Health During Life Years", style={'color':'dodgerblue','textAlign': 'center', 'borderRadius': '5px', 'boxShadow': '2px 2px 5px lightgrey'}),
                
                html.Label('Select Heart Status:'),
                dcc.Dropdown(
                    id='heart-disease-dropdown-left',
                    options=[
                        {'label': 'All Data', 'value': 'All'},
                        {'label': 'With Heart Disease', 'value': 'Yes'},
                        {'label': 'Without Heart Disease', 'value': 'No'}
                    ],
                    value='All',  # Default value
                    style={'width': '50%'}
                ),
                
                dcc.Graph(id='line-chart-left'),

                html.H3("BMI and Age Categories vs. Heart Disease", style={'color':'dodgerblue', 'textAlign': 'center', 'borderRadius': '5px', 'boxShadow': '2px 2px 5px lightgrey'}),
                html.Label('Select Gender:'),
                dcc.Dropdown(
                    id='sex-dropdown',
                    options=[
                        {'label': 'All Data', 'value': 'All'},
                        {'label': 'Male', 'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'}
                    ],
                    value='All',  # Default value
                    style={'width': '50%'}
                ),

                html.Label('Select Physical Activity:'),
                dcc.Dropdown(
                    id='activity-dropdown',
                    options=[
                        {'label': 'All Data', 'value': 'All'},
                        {'label': 'Doing Physical Activities', 'value': 'Yes'},
                        {'label': 'No Physical Activities', 'value': 'No'}
                    ],
                    value='All',  # Default value
                    style={'width': '50%'}
                ),

                html.Div(id='scatter-plot-container')
            ], className='six columns', style={'height': '500px', 'overflowY': 'scroll', 'border': '1px solid lightblue', 'border-radius': '5px', 'box-shadow': '2px 2px 15px lightblue'}),

        html.H2("Lifestyle Habits", style={'textAlign': 'center'}), 
        
            html.Div([
                html.H3("Percentage Distribution of Smoking and Alcohol Drinking Habits", style={'color':'dodgerblue','textAlign': 'center', 'borderRadius': '5px', 'boxShadow': '2px 2px 5px lightgrey'}),
                
                html.Label("Select Smoking and Drinking Habits:"),
                dcc.Dropdown(
                    id='label-dropdown-right',
                    options=[
                        {'label': 'No Smokers or Alcoholics', 'value': 'No Smokers or Alcoholics'},
                        {'label': 'Smokers Only', 'value': 'Smokers Only'},
                        {'label': 'Alcoholics Only', 'value': 'Alcoholics Only'},
                        {'label': 'Both', 'value': 'Both'}
                    ],
                    value='No Smokers or Alcoholics'
                ),
                
                html.Label("Select Health Status:"),
                dcc.Dropdown(
                    id='health-dropdown-right',
                    options=[
                        {'label': 'All Data', 'value': 'All'},
                        {'label': 'Very Good', 'value': 'Very good'},
                        {'label': 'Good', 'value': 'Good'},
                        {'label': 'Fair', 'value': 'Fair'},
                        {'label': 'Poor', 'value': 'Poor'}
                    ],
                    value='All'
                ),
                
                dcc.Graph(id='donut-chart-right'),
                
                html.H3("Sleep Time Habits", style={'color':'dodgerblue','textAlign': 'center', 'borderRadius': '5px', 'boxShadow': '2px 2px 5px lightgrey'}),
                
                html.Label('Select Heart Status:'),
                dcc.Dropdown(
                    id='heart-disease-dropdown-right',
                    options=[
                        {'label': 'All Data', 'value': 'All'},
                        {'label': 'With Heart Disease', 'value': 'Yes'},
                        {'label': 'Without Heart Disease', 'value': 'No'}
                    ],
                    value='All',  # Default value
                    style={'width': '50%'}
                ),
                
                dcc.Graph(id='line-chart-right'),
            ], className='six columns', style={'height': '500px', 'overflowY': 'scroll', 'border': '1px solid lightblue', 'border-radius': '5px', 'box-shadow': '2px 2px 5px lightblue'}),
        
        html.H2("Various Diseases Effect", style={'textAlign': 'center'}), 
        
            html.Div([
                dcc.Dropdown(
                    id='disease-dropdown',
                    options=[{'label': disease, 'value': disease} for disease in diseases_columns],
                    value=diseases_columns[0],
                    style={'width': '50%'}
                ),
            dcc.Graph(id='heatmap-graph'),
            ], className='six columns', style={'height': '500px', 'overflowY': 'scroll', 'border': '1px solid lightblue', 'border-radius': '5px', 'box-shadow': '2px 2px 15px lightblue'}),

         html.H2("Race Effect", style={'textAlign': 'center'}),
                  
            html.Div([
                html.H3("Heart Disease Distribution by Race", style={'color':'dodgerblue','textAlign': 'center', 'borderRadius': '5px', 'boxShadow': '2px 2px 5px lightgrey'}),

                dcc.Graph(id='bar-chart', figure=fig1),

                html.H3("Heart Disease (Males - Females) Distribution by Race", style={'color':'dodgerblue', 'textAlign': 'center', 'borderRadius': '5px', 'boxShadow': '2px 2px 5px lightgrey'}),

                dcc.Graph(id='bar-chart', figure=fig2),
            ], className='six columns', style={'height': '500px', 'overflowY': 'scroll', 'border': '1px solid lightblue', 'border-radius': '5px', 'box-shadow': '2px 2px 15px lightblue'}),
             
        ])
    ])
])

# Define callback to handle input and print the values
@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [
        dash.dependencies.State('input-race', 'value'),
        dash.dependencies.State('input-sex', 'value'),
        dash.dependencies.State('input-age-category', 'value'),
        dash.dependencies.State('input-bmi', 'value'),
        dash.dependencies.State('input-mental-health', 'value'),
        dash.dependencies.State('input-physical-health', 'value'),
        dash.dependencies.State('input-sleep-time', 'value'), 
        dash.dependencies.State('input-physical-activity', 'value'),
        dash.dependencies.State('input-smoking', 'value'),
        dash.dependencies.State('input-alcohol', 'value'),
        dash.dependencies.State('input-stroke', 'value'),
        dash.dependencies.State('input-diabetic', 'value'),
        dash.dependencies.State('input-asthma', 'value'),
        dash.dependencies.State('input-kidney-disease', 'value'),
        dash.dependencies.State('input-skin-cancer', 'value'),
        dash.dependencies.State('input-diff-walking', 'value'),
        dash.dependencies.State('input-general-health', 'value'),
        
    ]
)
def update_output(n_clicks, race_value, sex_value, age_category_value, bmi_value, 
                  mental_health_value, physical_health_value,sleep_time_value, 
                  physical_activity_value, smoking_value, alcohol_value,
                  stroke_value, diabetic_value, asthma_value, kidney_disease_value,
                  skin_cancer_value,diff_walking_value,general_health_value):
    
    print(f"n_clicks: {n_clicks} (type: {type(n_clicks)})")
    print(f"Race Value: {race_value} (type: {type(race_value)})")
    print(f"Sex Value: {sex_value} (type: {type(sex_value)})")
    print(f"Age Category Value: {age_category_value} (type: {type(age_category_value)})")
    print(f"BMI Value: {bmi_value} (type: {type(bmi_value)})")
    print(f"Mental Health Value: {mental_health_value} (type: {type(mental_health_value)})")
    print(f"Physical Health Value: {physical_health_value} (type: {type(physical_health_value)})")
    print(f"Sleep Time Value: {sleep_time_value} (type: {type(sleep_time_value)})")
    print(f"Physical Activity Value: {physical_activity_value} (type: {type(physical_activity_value)})")
    print(f"Smoking Value: {smoking_value} (type: {type(smoking_value)})")
    print(f"Alcohol Value: {alcohol_value} (type: {type(alcohol_value)})")
    print(f"Stroke Value: {stroke_value} (type: {type(stroke_value)})")
    print(f"Diabetic Value: {diabetic_value} (type: {type(diabetic_value)})")
    print(f"Asthma Value: {asthma_value} (type: {type(asthma_value)})")
    print(f"Kidney Disease Value: {kidney_disease_value} (type: {type(kidney_disease_value)})")
    print(f"Skin Cancer Value: {skin_cancer_value} (type: {type(skin_cancer_value)})")
    print(f"Difficulty Walking Value: {diff_walking_value} (type: {type(diff_walking_value)})")
    print(f"General Health Value: {general_health_value} (type: {type(general_health_value)})")
    
    
    
    if n_clicks > 0 and bmi_value is not None:
        # Create a DataFrame from the input values
        
        features = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth',
            'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity',
            'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer']
        data = pd.DataFrame([[bmi_value, smoking_value,alcohol_value, stroke_value, physical_health_value, mental_health_value,
                              diff_walking_value, sex_value, age_category_value, race_value, diabetic_value, physical_activity_value,
                              general_health_value, sleep_time_value, asthma_value, kidney_disease_value, skin_cancer_value]],
                            columns=features)
        features_to_scale_robust = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']


        data[features_to_scale_robust] = robust_scaler.transform(data[features_to_scale_robust])
        data_final = scaler.transform(data)
        print(data_final)
        # Make the prediction
        prediction = model.predict(data_final)
        print(prediction)
        
        # Display the prediction result
        if prediction == 1:
            result = "High Risk of Heart Disease"
        else:
            result = "Low Risk of Heart Disease"
        
       # Define style for the prediction result container
        result_style = {
            'border': '2px solid #007BFF',  # Border style with blue color
            'padding': '20px',  # Padding inside the container
            'margin-top': '20px',  # Top margin for spacing
            'font-size': '24px',  # Font size for the text
            'border-radius': '5px',
            'background-color': '#E3F2FD'
        }
        
        # Create a div containing prediction result with custom style
        prediction_div = html.Div([
            html.H3("Heart Disease Risk Prediction Result:", style={'color': 'Black'}),  # Heading with blue color
            html.P(result, style={'font-size': '22px', 'margin-top': '10px'})  # Paragraph with increased font size
        ], style=result_style)
        
        return prediction_div
    else:
        return ""


# Callback to update the line chart on the left based on the selected heart disease status
@app.callback(
    Output('line-chart-left', 'figure'),
    Input('heart-disease-dropdown-left', 'value')
)
def update_line_chart_left(selected_heart_disease):
    # Filter the data based on the selected heart disease status
    if selected_heart_disease == 'All':
        filtered_df = df
    else:
        filtered_df = df[df['HeartDisease'] == selected_heart_disease]
    
    # Group the data by AgeCategory and calculate the mean of PhysicalHealth and MentalHealth
    grouped_df = filtered_df.groupby('AgeCategory')[['PhysicalHealth', 'MentalHealth']].mean().reset_index()
    
    # Create line chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped_df['AgeCategory'], y=grouped_df['PhysicalHealth'], mode='lines', name='Physical Health'))
    fig.add_trace(go.Scatter(x=grouped_df['AgeCategory'], y=grouped_df['MentalHealth'], mode='lines', name='Mental Health'))
    
    # Update layout
    fig.update_layout(title=f'The effect of physical and mental health on heart disease during aging (Status:{selected_heart_disease})',
                      xaxis=dict(title='Age Category'),
                      yaxis=dict(title='Mean Health Score'))
    
    return fig

# Callback to generate the scatter plot on the left based on dropdown selections
@app.callback(
    Output('scatter-plot-container', 'children'),
    [Input('sex-dropdown', 'value'),
     Input('activity-dropdown', 'value')]
)
def update_scatter_plot(sex, activity):
    # Filter the dataset based on selected sex and physical activity
    if sex == 'All':
        filtered_df = df
    else:
        filtered_df = df[df['Sex'] == sex]

    if activity != 'All':
        filtered_df = filtered_df[df['PhysicalActivity'] == activity]
    
    # Scatter plot with BMI on x-axis, AgeCategory on y-axis, color-coded by presence of heart disease
    scatter_fig = px.scatter(filtered_df, x='BMI', y='AgeCategory', color='HeartDisease',
                             color_discrete_map={'Yes': 'red', 'No': 'cornflowerblue'},
                             title=f'BMI and Age Categories effect on Heart Disease (Gender: {sex}, Activity: {activity})',
                             labels={'BMI': 'BMI', 'AgeCategory': 'Age Category', 'HeartDisease': 'Heart Disease'})
    
    # Return the scatter plot component
    return dcc.Graph(id='bmi-age-scatter', figure=scatter_fig)

# Callback to update the donut chart on the right based on dropdown selections
@app.callback(
    Output('donut-chart-right', 'figure'),
    [Input('label-dropdown-right', 'value'),
     Input('health-dropdown-right', 'value')]
)
def update_donut_chart_right(label, health):
    filtered_df = df.copy()
    
    # Filter data based on health status
    if health != 'All':
        filtered_df = filtered_df[filtered_df['GenHealth'] == health]
    
    # Calculate the total count of individuals
    total_count = len(filtered_df)
    
    # Define labels for the selected option
    labels = {
        'No Smokers or Alcoholics': ['No', 'No'],
        'Smokers Only': ['Yes', 'No'],
        'Alcoholics Only': ['No', 'Yes'],
        'Both': ['Yes', 'Yes']
    }
    
    # Filter data based on label selection
    filtered_df = filtered_df[(filtered_df['Smoking'] == labels[label][0]) & (filtered_df['AlcoholDrinking'] == labels[label][1])]
    
    # Count the number of occurrences for each combination of smoking and alcohol drinking habits
    habit_counts = filtered_df.groupby('HeartDisease').size().reset_index(name='Count')
    
    # Calculate the percentage for each category
    habit_counts['Percentage'] = habit_counts['Count'] / total_count * 100
    
    # Generate the donut chart with an exploded layout
    donut_fig = px.pie(habit_counts, values='Percentage', names='HeartDisease', 
                       title=f'Percentage Distribution of {label} by Heart Status',
                       hole=0.3,
                       template='plotly',
                       color_discrete_sequence=['cornflowerblue', 'red']
                       )
    donut_fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                  marker=dict(line=dict(color='#000000', width=2)))
    donut_fig.update_layout(legend_title_text='Heart Disease')
    donut_fig.update_layout(showlegend=True)
    donut_fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                  marker=dict(line=dict(color='#000000', width=2)),
                  pull=[0.1, 0])
    
    return donut_fig

# Callback to update the line chart on the right based on the selected heart disease status
@app.callback(
    Output('line-chart-right', 'figure'),
    Input('heart-disease-dropdown-right', 'value')
)
def update_line_chart_right(selected_heart_disease):
    # Filter the data based on the selected heart disease status
    if selected_heart_disease == 'All':
        filtered_df = df
    else:
        filtered_df = df[df['HeartDisease'] == selected_heart_disease]
    
    # Group the data by AgeCategory and calculate the mean of SleepTime
    grouped_df = filtered_df.groupby('AgeCategory')['SleepTime'].mean().reset_index()
    
    # Create line chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped_df['AgeCategory'], y=grouped_df['SleepTime'], mode='lines', name='Sleep Time'))
    
    # Update layout
    fig.update_layout(title=f'The Relation Between Sleep Time and Heart Disease During Aging (Status:{selected_heart_disease})',
                      xaxis=dict(title='Age Category'),
                      yaxis=dict(title='Sleep Time'))
    
    return fig

# Callback to update the heatmap based on changes in the dropdown menu
@app.callback(
    Output('heatmap-graph', 'figure'),
    Input('disease-dropdown', 'value')
)
def update_heatmap(selected_disease):
    # Retrieve the contingency table for the selected disease
    heatmap_data = contingency_tables[selected_disease]
    
    # Define a custom color scale with different colors
    custom_color_scale = [
        (0.0, 'DodgerBlue'), 
        (0.2, 'LightSkyBlue'), 
        (0.4, 'LightGreen'), 
        (0.6, 'Gold'), 
        (0.8, 'Tomato'), 
        (1.0, 'FireBrick')
    ]
    
    # Create heatmap using Plotly Express with custom color scale
    heatmap_fig = px.imshow(heatmap_data, 
                            labels=dict(x="Heart Disease", y="Disease", color="Frequency"),
                            x=heatmap_data.columns,
                            y=heatmap_data.index,
                            color_continuous_scale=custom_color_scale)
    
    # Update layout
    heatmap_fig.update_layout(title=f"Relationship Between {selected_disease} and Heart Disease",
                              xaxis=dict(title="Heart Disease"),
                              yaxis=dict(title="Disease"))
    
    return heatmap_fig

# Run the app
if __name__ == '__main__':
    app.run_server()


# %%



