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
from datetime import date
from sklearn import preprocessing
import dash_bootstrap_components as dbc
import modules.fred_scraper as fred_scraper
import os


# external css static
external_stylesheets = [dbc.themes.SLATE]

# dash app with external stylesheet
app = Dash(__name__, external_stylesheets=external_stylesheets)


# Import and clean data
try:
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
days = [i for i in range(0, 13574)]
df_dates = [i.date() for i in df.index]

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div(
    [
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
            ],
            value=["ECOGROWTH"],
            labelStyle={"fontSize": "20px", "margin-right": "5px"},
        ),
        html.Br(),
        dcc.Graph(
            id="history_plot",
            figure={},
            style={"width": "100%", "height": "67vh", "backgroundColor": "black"},
        ),
        html.Br(),
        html.Br(),
        html.H1(
            "Visualization Tool for Past Economic Cycles",
            style={"text-align": "center"},
        ),
        html.Br(),
        dcc.Dropdown(
            id="select_cycle",
            options=[
                {"label": "1975-1982", "value": 0},
                {"label": "1982-1992", "value": 1},
                {"label": "1992-2002", "value": 2},
                {"label": "2002-2009", "value": 3},
                {"label": "2009-2020", "value": 4},
            ],
            multi=False,
            value=4,
            style={"width": "40%", "fontsize": "25px"},
        ),
        html.Br(),
        dcc.Graph(
            id="compare_plot", figure={}, style={"width": "100%", "height": "67vh"}
        ),
        html.Br(),
        html.Br(),
        html.H1("Comparison with slider", style={"text-align": "center"}),
        html.Br(),
        html.Div(
            dcc.RangeSlider(
                id="select_year",
                marks={
                    i: {
                        "label": "   " + str(df_dates[i]),
                        "style": {"transform": "rotate(45deg)", "color": "white"},
                    }
                    for i in range(0, len(days), 200)
                },
                min=0,
                max=len(days) - 1,
                value=[0, len(days) - 1],
            ),
            style={
                "margin-left": "50px",
                "margin-right": "50px",
                "margin-top": "25px",
                "margin-bottom": "25px",
            },
        ),
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
# Connect the Plotly graphs with Dash Components
@app.callback(
    [
        Output(component_id="output_container", component_property="children"),
        Output(component_id="history_plot", component_property="figure"),
        Output(component_id="compare_plot", component_property="figure"),
        Output(component_id="compare_slider_plot", component_property="figure"),
    ],
    [
        Input(component_id="select_variables", component_property="value"),
        Input(component_id="select_cycle", component_property="value"),
        Input(component_id="select_year", component_property="value"),
    ],
)
def update_graph(variables, cycle, slider):
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

    # print variables selection
    container = f"The variables chosen by user was: {variables}, the cycle chosen was {bottoms[cycle]} to {bottoms[cycle+1]}"

    # --------------------------------
    # first graph: historical data
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

    # -------------------------------
    # second graph: cycles visualization and comparison

    dff = df.copy()

    # loc data on right cycle
    past_cycle = dff.loc[(dff.index > bottoms[cycle]) & (df.index < bottoms[cycle + 1])]
    present_cycle = dff.loc[(dff.index > bottoms[-2])]
    index = [i for i in past_cycle.index]

    comparison = go.Figure(layout=layout)

    # for column in callback plot it
    for column in variables:
        # origin = 0
        past_cycle[column] = past_cycle[column] - past_cycle[column][0]
        present_cycle[column] = present_cycle[column] - present_cycle[column][0]
        # smoothen
        past_cycle[column] = past_cycle[column].rolling(window=45).mean()
        present_cycle[column] = present_cycle[column].rolling(window=45).mean()

        # Plotly Express
        comparison.add_trace(
            go.Scatter(
                x=index,
                y=past_cycle[column],
                mode="lines",
                name=f"{column} past",
                opacity=0.7,
            )
        )
        comparison.add_trace(
            go.Scatter(
                x=index,
                y=present_cycle[column],
                mode="lines",
                name=f"{column}  present",
                opacity=0.7,
            )
        )

    # update layout with new title
    comparison.update_layout(
        title={
            "text": "Economic cycles comparison (present vs past)",
            "font": {"size": 24},
        }
    )

    # --------------------------------------
    # third graph: comparison on date range

    dff = df.copy()

    index = dff.index

    past_cycle = dff.loc[
        (dff.index > dff.index[slider[0]]) & (df.index < dff.index[slider[1]])
    ]
    present_cycle = dff.loc[(dff.index > bottoms[-2])]

    index = [i for i in past_cycle.index]

    comparison_slider = go.Figure(layout=layout)
    for column in variables:
        # origin = 0
        past_cycle[column] = past_cycle[column] - past_cycle[column][0]
        present_cycle[column] = present_cycle[column] - present_cycle[column][0]
        # smoothen
        past_cycle[column] = past_cycle[column].rolling(window=45).mean()
        present_cycle[column] = present_cycle[column].rolling(window=45).mean()

        # Plotly Express
        comparison_slider.add_trace(
            go.Scatter(
                x=index,
                y=past_cycle[column],
                mode="lines",
                name=f"{column} past",
                opacity=0.7,
            )
        )
        comparison_slider.add_trace(
            go.Scatter(
                x=index,
                y=present_cycle[column],
                mode="lines",
                name=f"{column}  present",
                opacity=0.7,
            )
        )

    # update layout with new title
    comparison_slider.update_layout(
        title={
            "text": "Economic cycles comparison_slider (present vs past)",
            "font": {"size": 24},
        }
    )

    return container, hist, comparison, comparison_slider


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
