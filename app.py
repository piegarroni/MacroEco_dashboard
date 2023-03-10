import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    ctx,
)  # pip install dash (version 2.0.0 or higher)
from datetime import date, datetime
from sklearn import preprocessing
import dash_bootstrap_components as dbc
import modules.fred_scraper as fred_scraper
import os
import numpy as np



# external css static
external_stylesheets = [dbc.themes.SLATE]

# dash app with external stylesheet
app = Dash(__name__, external_stylesheets=external_stylesheets)


# Import and clean data
try:
    print('data just downloaded')
    my_instance = fred_scraper.fred_retriever()
    df = my_instance.run()
    df.to_csv(os.getcwd() + "/data.csv")
    index = df.index
except:
    print("failed to download data")
    df = pd.read_csv("data.csv")
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True)
    index = df["DATE"]
    df = df.set_index("DATE")

print(df.columns)
# bottoms economy
bottoms = [
    "04-01-1975",
    "12-01-1982",
    "03-01-1992",
    "08-01-2002",
    "03-01-2009",
    "05-01-2020",
    "02-01-2021",
    str(date.today()),
]



# days ranges
start_date = "2000-01-22"
end_date  = datetime.today().strftime('%Y-%m-%d')
date_range = pd.date_range(start=start_date, end=end_date, freq="B")


# Define graphs layout
layout = go.Layout(
    title={"text": "Economy history", "font": {"size": 24}},
    xaxis={"title": "Date", "titlefont": {"size": 20}, "tickfont": {"size": 16}},
    yaxis={
        "title": "Scaled Value",
        "titlefont": {"size": 20},
        "tickfont": {"size": 16},
    },
    plot_bgcolor="#d7d9e0",
)

def preprocess_quadrants(df):
    # growth

    dff = df.copy()

    dff['GDP'] = pd.to_numeric(dff['GDP'])
    # Calculate inflation rate
    dff['GDP_rate'] = (dff['GDP'] - dff['GDP'].shift(1)) / dff['GDP'].shift(1)

    # Convert inflation rate to percentage
    dff['GDP_rate']  = dff['GDP_rate']  * 100


    # Smoothen it
    dff['GDP_rate'] = dff['GDP_rate'].rolling(4).mean() - 0.745

    dff['GDP_rate_log'] = np.log10(dff['GDP_rate'])
    dff['GDP_rate_log'] =dff['GDP_rate_log'].fillna(-400)

    #inflation

    dff['CPIAUCSL'] = pd.to_numeric(dff['CPIAUCSL'])
    # Calculate inflation rate
    dff['CPIAUCSL_rate'] = (dff['CPIAUCSL'] - dff['CPIAUCSL'].shift(1)) / dff['CPIAUCSL'].shift(1)

    # Convert inflation rate to percentage
    dff['CPIAUCSL_rate'] = dff['CPIAUCSL_rate'] * 1000

  
    # Smoothen it

    dff['CPIAUCSL_rate'] = dff['CPIAUCSL_rate'].rolling(24).mean() -1.9

    #log 
    dff['CPIAUCSL_rate_log'] = np.log10(-dff['CPIAUCSL_rate'])
    dff['CPIAUCSL_rate_log'] = dff['CPIAUCSL_rate_log'].fillna(-400)



    quadrant = []
    for i in range(len(dff)):
        if dff['GDP_rate_log'][i] != -400 and dff['CPIAUCSL_rate_log'][i] == -400:
            quadrant.append(1)
            #print('invest in ("Quadrant 1: Inflation up, GDP up")') #quadrant 1 consists of the following assets= ["Emerging equities", "International real estate", "Gold", "Commodities", "emerging bond spreads", "Inflation protected bonds"]  
        elif dff['GDP_rate_log'][i] == -400 and dff['CPIAUCSL_rate_log'][i] == -400:
            quadrant.append(2)

            #print('invest in ("Quadrant 2: Inflation up, GDP down")') # quadrant 2 consists of the following assets= ["Gold", "Commodities", "Emerging bond spreads", "Inflation protected bonds", "cash"]
        elif dff['GDP_rate_log'][i] == -400 and dff['CPIAUCSL_rate_log'][i] != -400:
            quadrant.append(3)
            #print('invest in ("Quadrant 3: Inflation down, GDP up")') # quadrant 3 consists of the following assets= ["Developed corporate bond spreads", "intermediate treasuries", "Developed real estate", "Developed equities"]  
        elif dff['GDP_rate_log'][i] != -400 and dff['CPIAUCSL_rate_log'][i] != -400: # add if to remember state
            quadrant.append(4)
            #print('invest in ("Quadrant 4: Inflation down, GDP down")') #quadrant 4 consists of the following assets = ["Gold", "Long term treasuries", "cash"]
        else:
            quadrant.append(np.nan)

  

    # clean quadrants
    quadrants = [i for i in  quadrant]
    for i in range(1, len(quadrants)):
        if quadrants[i] ==4 and quadrants[i-1] == 1:
            quadrants[i]=1
        

    dff['quadrant_cleaned'] = quadrants

    return dff['quadrant_cleaned']

  

def visualize_quadrants(series):
    """
    Function to visualize the macroeconomical quadratns
    """
    
    fig = go.Figure(layout=layout)
  
    fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines')) ###  x, y ????
    fig.update_layout(title='Economical quadrant history',
                      xaxis_title='Date',
                      yaxis_title='Quadrant')
    return fig



def visualize_history(variables):
    """
    Function to visualize the Relative Rotation Graph of the selected symbol
    """

    dff = df.copy()
    hist = go.Figure(layout=layout)
    for column in variables:
        # smoothen
        dff[column] = dff[column].rolling(window=45).mean()
        # Plotly Express
        hist.add_trace(
            go.Scatter(
                x=dff.index, y=dff[column], mode="lines", name=column, opacity=0.7
            )
        )

    return hist


def visualize_compare(variables, timerange):
    dff = df.copy()

    index = dff.index

    past_cycle = dff.loc[
        (dff.index > dff.index[timerange[0]]) & (df.index < dff.index[timerange[1]])
    ]
    present_cycle = dff.loc[(dff.index > bottoms[-2])]

    index = [i for i in past_cycle.index]

    fig =  go.Figure(layout=layout)
    for column in variables:
        # set origin = 0
        past_cycle[column] = past_cycle[column] - past_cycle[column][0]
        present_cycle[column] = present_cycle[column] - present_cycle[column][0]
        # smoothen variables
        past_cycle[column] = past_cycle[column].rolling(window=45).mean()
        present_cycle[column] = present_cycle[column].rolling(window=45).mean()

        fig.add_trace(
            go.Scatter(
                x=index,
                y=past_cycle[column],
                mode="lines",
                name=f"{column} past",
                opacity=0.7,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index,
                y=present_cycle[column],
                mode="lines",
                name=f"{column}  present",
                opacity=0.7,
            )
        )

    # update layout with new title
    fig.update_layout(
        title={
            "text": "Economic cycles comparison (present vs past)",
            "font": {"size": 24},
        }
    )
    return fig


# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div(
    [
        # ----------------------------------------------------------- first graph
        html.H1(
            "Visualization Tool for History Economy", style={"text-align": "center"}
        ),
        dcc.Checklist(
            id="select_variables",
            options=[
                {"label": "  Interest rates  ", "value": "DFF"},
                {"label": "  Growth  ", "value": "ECOGROWTH"},
                {"label": "  Inflation  ", "value": "INFLATION"},
                {"label": "  Nasdaqcom  ", "value": "NASDAQCOM"},
                {"label": "  Nyse ", "value": "BOGZ1FL073164003Q"},
                {"label": "  Willshire ", "value": "WILL5000INDFC"},
                {"label": "  5y Breakeven Inflation", "value": "T5YIE"},
                {"label": "  GDP", "value": "GDP"},
                {"label": "  CPI", "value": "CPIAUCSL"},

            ],
            value=["ECOGROWTH"],
            labelStyle={"fontSize": "20px", "margin-right": "5px"},
        ),
        html.Div(
            dcc.RangeSlider(
                id="select_year",
                marks={
                    i: {
                        "label": "   " + str(date_range[i]),
                        "style": {"transform": "rotate(45deg)", "color": "white"},
                    }
                    for i in range(0, len(date_range), 200)
                },
                min=0,
                max=len(date_range) - 1,
                value=[0, len(date_range) - 1],
            ),
            style={
                "margin-left": "50px",
                "margin-right": "50px",
                "margin-top": "25px",
                "margin-bottom": "25px",
            },
        ),
        html.Br(),

        dcc.Graph(
            id="history_plot",
            figure={},
            style={"width": "100%", "height": "67vh", "backgroundColor": "black"},
        ),
        html.Br(),
        html.Br(),


        # ----------------------------------------------------------------------- second graph
        html.H1(
            "4 Quadrants visualization history",
            style={"text-align": "center"},
        ),
        html.Br(),
       
        dcc.Graph(
            id="quadrants", figure={}, style={"width": "100%", "height": "67vh"}
        ),
        html.Br(),
        html.Br(),


        # ---------------------------------------------------------------------- third graph
        html.H1("Compare current cycles with historical data", style={"text-align": "center"}),
        html.Br(),
      
        html.Div(id="output"),
        html.Br(),
        dcc.Graph(
            id="compare_slider_plot",
            figure={},
            style={"width": "100%", "height": "67vh"},
        ),
        html.Br(),
        html.Br(),
        html.Div(id="output_container", children=[]),
    ],
    style={"margin": "35px 5% 75px 5%"},
)


# ------------------------------------------------------------------------------
@app.callback(
    [
        Output(component_id="output_container", component_property="children"),
        Output(component_id="history_plot", component_property="figure"),
        Output(component_id="quadrants", component_property="figure"),
        Output(component_id="compare_slider_plot", component_property="figure"),
    ],
    [
        Input(component_id="select_variables", component_property="value"),
        Input(component_id="select_year", component_property="value"),
    ],
)
def update_graph(variables, slider):

    container = f"The variables chosen by user was: {variables}"

    return container, visualize_history(variables), visualize_quadrants(preprocess_quadrants(df)), visualize_compare(variables, slider)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
