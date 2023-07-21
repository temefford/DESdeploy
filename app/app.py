import numpy as np
import pandas as pd
from time import sleep

import dash 
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash import html
from dash import dcc
from dash import dash_table
from dash import Dash
import dash_bootstrap_components as dbc
import plotly as py
import plotly.express as px
import plotly.tools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import dash_canvas
from dash_canvas.components import image_upload_zone
from dash_canvas.utils import (
    image_string_to_PILImage,
    array_to_data_url,
    parse_jsonstring_line,
    brightness_adjust,
    contrast_adjust,
)
import pathlib
import utils as vru

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
# get relative data folder
PATH = pathlib.Path(__file__).parent

DATA_PATH = PATH.joinpath("data").resolve()

table_header_style = {
    "backgroundColor": "rgb(2,21,70)",
    "color": "white",
    "textAlign": "center",
}

class simulation_parameters:
    sim_duration = 60*5
    num_rads = 6
    arr_rate_1 = 2
    arr_rate_2 = 2
    arr_rate_3 = 2
    proc_rate_1 = 2
    proc_rate_2 = 2
    proc_rate_3 = 2
    targ_time_1 = 30
    targ_time_2 = 60
    targ_time_3 = 24*60
    percent_special = .40
    cutoff = 2
    verbose = False
    fig = px.scatter()


def demo_explanation():
    # Markdown files
    with open(PATH.joinpath("demo.md"), "r") as file:
        demo_md = file.read()

    return html.Div(
        html.Div([dcc.Markdown(demo_md, className="markdown")]),
        style={"margin": "10px"},
    )

"""""
def run_simulation(): 
    print("Running sim")
    sim_duration = simulation_parameters.sim_duration
    num_rads = simulation_parameters.num_rads
    arr_rates = [simulation_parameters.arr_rate_1, simulation_parameters.arr_rate_2, simulation_parameters.arr_rate_3]
    proc_rates = [simulation_parameters.proc_rate_1, simulation_parameters.proc_rate_2, simulation_parameters.proc_rate_3]
    constant_rads = False
    cutoff = simulation_parameters.cutoff
    verbose = simulation_parameters.verbose

    sys_state = vru.sim(sim_duration, num_rads, arr_rates, proc_rates, constant_rads, cutoff, verbose)
    img_table = sys_state.img_table
    print("sim done")
    simulation_parameters.fig = px.scatter(img_table, x="time_created", y="total_time",
                                            color="urgency", hover_name="img_id",
                                            log_x=False, size_max=60)
"""""



def demo_data():
    #Define variables
    sim_duration = 60*5
    num_rads = 6
    arr_rate_var = 2
    arr_rates = [arr_rate_var, arr_rate_var, arr_rate_var]
    urg_times = [2, 5, 10]
    #run sim
    sys_state = vru.sim(sim_duration, num_rads, arr_rates, urg_times, cutoff=True, verbose=False)
    print("demo")


def instructions():
    return html.Div(children=[html.A("GitHub Repo" , href='https://github.com/temefford/DES', target="_blank"),
                              html.P(" "),
                              html.A("Google Colab", href='https://colab.research.google.com/drive/1G42XJu26m4uWVqzKP4lPmdGlcnXTQrU-#scrollTo=WTCWsDeImdxS', target="_blank"), 
                    ],
                    style={'font-size': '16px', 'display': 'inline-block', 'margin-left': '25px', 'margin-right': '5px', 'margin-bottom': '15px'},
                    )


height, width = 200, 500
canvas_width = 800
canvas_height = round(height * canvas_width / width)
scale = canvas_width / width


def div_graph(name):
    # Generates an html Div containing graph and control options for smoothing and display, given the name
    return html.Div(
        className="row",
        children=[
            html.Div(
                className="two columns",
                style={"padding-bottom": "5%"},
                children=[
                    html.Div(
                        [
                            html.P(
                                "Plot Display Mode:",
                                style={"font-weight": "bold", "margin-bottom": "0px"},
                                className="plot-display-text",
                            ),
                            html.Div(
                                [
                                    dcc.RadioItems(
                                        options=[
                                            {
                                                "label": " All",
                                                "value": "overlap",
                                            },
                                            {
                                                "label": " Urgent",
                                                "value": "separate_vertical",
                                            },
                                        ],
                                        value="overlap",
                                        id=f"radio-display-mode-{name}",
                                        labelStyle={"verticalAlign": "middle"},
                                        className="plot-display-radio-items",
                                    )
                                ],
                                className="radio-item-div",
                            ),
                            html.Div(id=f"div-current-{name}-value"),
                        ],
                        className="entropy-div",
                    ),
                ],
            ),
            html.Div(id=f"div-{name}-graph", className="ten columns"),
        ],
    )

app.layout = html.Div(
    children=[
        html.Div(
            [
                html.Img(
                    src=app.get_asset_url("usfca-logo.png"), className="usfca-logo", style={"margin-bottom": "10px","margin-top":"20px", "margin-left":"40px", "width":"30%"},
                ),
                html.H1(children="TeleRadiology Simulation", style={'fontSize': 40, 'padding': 3, 'margin-left': 25, 'margin-right': 5}),
                html.H5(children="This application simulates a network of remote radiologists analyzing medical images. It is in essence a queuing problem that can be used to explore routing algorithms (deciding who to send each new medical image to) to optimize the network's efficiency.", style={'color': 'white', 'fontSize': 15, 'padding': 3, 'margin-left': 25, 'margin-right': 5}),
                instructions(),
                html.Div(
                    [
                        html.Button(
                            "LEARN MORE",
                            className="button_instruction",
                            id="learn-more-button",
                            style={'horizontalAlign':'middle', 'verticalAlign':'middle'},
                        ),
                    ],
                    className="mobile_buttons",
                    style={'horizontalAlign':'middle', 'verticalAlign':'middle'},
                ),
                html.Div(
                    # Empty child function for the callback
                    html.Div(id="demo-explanation", children=[])
                ),
                html.Div(
                    [
                        html.H2("Adjust the System Parameters", style={'padding':10, 'margin-bottom': '5px'}),
                        html.Div(
                            [
                                html.Div([
                                    html.Label("Duration of Simulation (hrs)", style={'margin-bottom': '5px', 'margin-left': '25px'}),
                                    dcc.Input(
                                        id="sim-duration",
                                        type="number",
                                        value=5,
                                        name="simulation duration",
                                        min=1,
                                        max=10,
                                        step=1,
                                        style={'font-size': '12px', 'width': '80px', 'display': 'inline-block', 'margin-left': '45px', 'margin-right': '5px'}
                                    ),
                                ],
                                style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),

                                html.Div([
                                    html.Label("Number of Radiologists", style={'margin-bottom': '5px', 'margin-left': '5px'}),
                                    dcc.Input(
                                        id="num-rads",
                                        type="number",
                                        value=6,
                                        name="number of radiologists",
                                        min=1,
                                        step=1,
                                        style={'font-size': '12px', 'width': '80px', 'display': 'inline-block', 'margin-left': '25px', 'margin-right': '5px'}
                                    ),
                                ],
                                style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}
                                ),   
                            ],
                            style={'horizontalAlign':'middle', 'verticalAlign':'middle', 'width': '100%', 'display':'table'}),

                        html.Label("Time Between Image Generation (minutes)", style={'margin-bottom': '5px', 'margin-top': '45px'}),
                        html.Div(
                            [
                                html.Div([
                                    html.P("Urgency 1", style={'margin-left': '40px'}),
                                    dcc.Input(
                                        id="arr-rate-1",
                                        type="number",
                                        value=2,
                                        min=0,
                                        step=.1,
                                        style={'font-size': '12px', 'width': '80px', 'display': 'inline-block', 'margin-left': '25px', 'margin-right': '15px'}
                                    )
                                ],
                                style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),

                                html.Div([
                                    html.P("Urgency 2", style={'margin-left': '30px'}),
                                    dcc.Input(
                                        id="arr-rate-2",
                                        type="number",
                                        value=2,
                                        min=0,
                                        step=.1,
                                        style={'font-size': '12px', 'width': '80px', 'display': 'inline-block', 'margin-left': '15px', 'margin-right': '15px'}
                                    )
                                ],
                                style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),

                                html.Div([
                                    html.P("Urgency 3", style={'margin-left': '30px'}),
                                    dcc.Input(
                                        id="arr-rate-3",
                                        type="number",
                                        value=2,
                                        min=0,
                                        step=.1,
                                        style={'font-size': '12px', 'width': '80px', 'display': 'inline-block', 'margin-left': '15px', 'margin-right': '25px'}
                                    )
                                ],
                                style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),

                            ],
                            style={'horizontalAlign':'middle', 'verticalAlign':'middle', 'width': '100%', 'display':'table'}),
                                
                                
                        html.Label("Average Process Times for Images (minutes)", style={'margin-bottom': '5px', 'margin-top': '45px'}),
                        html.Div(
                            [
                                html.Div([
                                    html.P("Urgency 1", style={'margin-left': '40px'}),
                                    dcc.Input(
                                        id="proc-rate-1",
                                        type="number",
                                        value=2,
                                        min=0,
                                        step=.1,
                                        style={'font-size': '12px', 'width': '80px', 'display': 'inline-block', 'margin-left': '25px', 'margin-right': '15px'}
                                    )
                                ],
                                style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),

                                html.Div([
                                    html.P("Urgency 2", style={'margin-left': '30px'}),
                                    dcc.Input(
                                        id="proc-rate-2",
                                        type="number",
                                        value=2,
                                        min=0,
                                        step=.1,
                                        style={'font-size': '12px', 'width': '80px', 'display': 'inline-block', 'margin-left': '15px', 'margin-right': '15px'}
                                    )
                                ],
                                style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),

                                html.Div([
                                    html.P("Urgency 3", style={'margin-left': '30px'}),
                                    dcc.Input(
                                        id="proc-rate-3",
                                        type="number",
                                        value=2,
                                        min=0,
                                        step=.1,
                                        style={'font-size': '12px', 'width': '80px', 'display': 'inline-block', 'margin-left': '15px', 'margin-right': '25px'}
                                    )
                                ],
                                style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),

                            ],
                            style={'horizontalAlign':'middle', 'verticalAlign':'middle', 'width': '100%', 'display':'table'}
                            ),

                        html.Label("Set Target Time for Image Processing (minutes)", style={'margin-bottom': '5px', 'margin-top': '45px'}),
                        html.Div(
                            [
                                html.Div([
                                    html.P("Urgency 1", style={'margin-left': '40px'}),
                                    dcc.Input(
                                        id="targ-rate-1",
                                        type="number",
                                        value=30,
                                        min=0,
                                        step=.1,
                                        style={'font-size': '12px', 'width': '80px', 'display': 'inline-block', 'margin-left': '25px', 'margin-right': '15px'}
                                    )
                                ],
                                style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),

                                html.Div([
                                    html.P("Urgency 2", style={'margin-left': '30px'}),
                                    dcc.Input(
                                        id="targ-rate-2",
                                        type="number",
                                        value=60,
                                        min=0,
                                        step=.1,
                                        style={'font-size': '12px', 'width': '80px', 'display': 'inline-block', 'margin-left': '15px', 'margin-right': '15px'}
                                    )
                                ],
                                style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),

                                html.Div([
                                    html.P("Urgency 3", style={'margin-left': '30px'}),
                                    dcc.Input(
                                        id="targ-rate-3",
                                        type="number",
                                        value=60*24,
                                        min=0,
                                        step=.1,
                                        style={'font-size': '12px', 'width': '80px', 'display': 'inline-block', 'margin-left': '15px', 'margin-right': '25px'}
                                    )
                                ],
                                style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),

                            ],
                            style={'horizontalAlign':'middle', 'verticalAlign':'middle', 'width': '100%', 'display':'table'}
                            ),
                               
                    ],
                    className="mobile_forms",
                ),
                html.Div(
                    [
                        html.Label("Fraction of specialist radiologists to non-specialists)", style={'margin-bottom': '10px', 'margin-top': '45px', 'margin-right':'15px'}),
                        html.Div(
                            [
                                dcc.Slider(
                                min=0, 
                                max=1, 
                                step=0.05,
                                value=0.4,
                                id='percent-special',
                                marks={'0':'0', '0.2':'0.2', '0.4':'0.4', '0.6':'0.6', '0.8':'0.8', '1':'1'},
                                tooltip={"placement": "bottom", "always_visible": True})
                            ],
                            style={'width': '87%', "margin-left": "15px", "margin-right": "5px", "margin-top": "10px"}
                            ),
                        html.Div(
                            [
                                html.Div([
                                    html.Label("Verbose", style={"margin-left": "50px", 'margin-bottom': '5px', 'margin-top': '35px'}),
                                    dcc.Checklist(
                                    id="verbose-val",
                                    options=[{"label": "", "value": 1}],
                                    value=[1],
                                    style={"margin-left": "50px", "margin-right": "5px"},
                                    ),
                                ],
                                style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}
                                ),
                                html.Div([
                                    html.Label("Cutoff Time", style={"margin-left": "45px", 'margin-bottom': '0px', 'margin-top': '25px'}),
                                    html.Label("(mult of sim duration)", style={"margin-left": "20px", 'margin-bottom': '0px', 'margin-top': '0px'}),
                                    dcc.RadioItems(
                                        id="cutoff",
                                        options=[
                                            {"label": "1", "value": "1"},
                                            {"label": "2", "value": "2"},
                                            {"label": "4", "value": "4"},
                                            {"label": "8", "value": "8"},
                                        ],
                                        value="2",
                                        labelStyle={"display": "inline-block"},
                                        style={"margin-left": "-20px", "margin-right": "80px"},
                                        ),
                                    ],
                                    style={'display':'table-cell', 'margin-top': '0px', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}
                                    )   
                            ],
                            style={'horizontalAlign':'middle', 'verticalAlign':'middle', 'width': '100%', 'display':'table'}
                            ),
                        
                        
                    ],
                    className="radio_items",
                ),
                
                html.Button(
                    "Run Simulation", id="button-run-sim", className="button_submit", style={"margin-top": "50px", "margin-left": "110px"},
                ),
            ],
            className="four columns instruction",
        ),
        html.Div(
            className="eight columns result",
            children=[
                html.H1(children="Simulation Parameters", style={'color': 'black', 'fontSize': 30, 'padding': 2, 'margin-left': 50, 'margin-right': 5}),
                #html.H1(id="results-params", style={'color': 'black', 'fontSize': 10, 'padding': 2, 'margin-left': 50, 'margin-right': 5}),
                html.Div([
                    html.Div([
                                html.H4("Duration of Simulation (hrs)", style={'margin-bottom': '25px', 'margin-left': '25px'}),
                                html.H6(
                                    id="sim-dur",
                                    style={'color':'blue', 'margin-top': '45px', 'margin-bottom': '5px', 'margin-left': '40px'},
                                ),
                            ],
                            style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),
                    #html.Div([html.H6("Simulation Duration (minutes):"), html.H6(id="sim-dur")]),
                    html.Div([
                                html.H4("Number of Radiologists", style={'margin-bottom': '25px', 'margin-left': '15px'}),
                                html.H6(
                                    id="num-rads-o",
                                    style={'color':'blue', 'margin-top': '45px', 'margin-bottom': '5px', 'margin-left': '40px'},
                                ),
                            ],
                            style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),
                    html.Div([
                                html.H4("Time Between Images", style={'margin-bottom': '5px', 'margin-left': '5px'}),
                                html.H6(
                                    id="arr-rate-1-o",
                                    style={'color':'red', 'margin-bottom': '5px', 'margin-left': '10px'},
                                ),
                                html.H6(
                                    id="arr-rate-2-o",
                                    style={'color':'orange', 'margin-bottom': '5px', 'margin-left': '10px'},
                                ),
                                html.H6(
                                    id="arr-rate-3-o",
                                    style={'color':'green', 'margin-bottom': '5px', 'margin-left': '10px'},
                                ),
                            ],
                            style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),
                    html.Div([
                                html.H4("Avg Process Time Once Seen", style={'margin-bottom': '5px', 'margin-left': '5px'}),
                                html.H6(
                                    id="proc-rate-1-o",
                                    style={'color':'red', 'margin-bottom': '5px', 'margin-left': '10px'},
                                ),
                                html.H6(
                                    id="proc-rate-2-o",
                                    style={'color':'orange', 'margin-bottom': '5px', 'margin-left': '10px'},
                                ),
                                html.H6(
                                    id="proc-rate-3-o",
                                    style={'color':'green', 'margin-bottom': '5px', 'margin-left': '10px'},
                                ),
                            ],
                            style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),
                    html.Div([
                                html.H4("Time Limit for Image to be Processed", style={'margin-bottom': '5px', 'margin-left': '5px'}),
                                html.H6(
                                    id="targ-rate-1-o",
                                    style={'color':'red', 'margin-bottom': '5px', 'margin-left': '10px'},
                                ),
                                html.H6(
                                    id="targ-rate-2-o",
                                    style={'color':'orange', 'margin-bottom': '5px', 'margin-left': '10px'},
                                ),
                                html.H6(
                                    id="targ-rate-3-o",
                                    style={'color':'green', 'margin-bottom': '5px', 'margin-left': '10px'},
                                ),
                            ],
                            style={'display':'table-cell', 'padding':0, 'horizontalAlign':'middle', 'verticalAlign':'middle'}),
                ]
                ),
                html.H1(children="Plots", style={'color': 'black', 'fontSize': 30, 'padding': 2, 'margin-left': 50, 'margin-right': 5}),
                dcc.Graph(id="results-graph"), 
                #html.Img(id="idl_plot", src='children', height=300),
                
                dcc.Graph(id="idl-plot"),
                dcc.Graph(id="busy-plot"),
                html.H1(children="Simulation Results", style={'color': 'black', 'fontSize': 30, 'padding': 2, 'margin-left': 50, 'margin-right': 5}),
                html.H1(children="*All times in minutes*", style={'color': 'grey', 'fontSize': 15, 'padding': 2, 'margin-left': 50, 'margin-right': 5}),
                dash_table.DataTable(
                    id="results-table",
                    style_header=table_header_style,
                    style_data_conditional=[
                        {
                            "if": {"column_id": "param"},
                            "textAlign": "right",
                            "paddingRight": 15,
                        },
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "white",
                        },
                    ],
                    page_size=10,
                    page_action='none',
                    style_table={'width':'80%', 'horizontalAlign':'middle', 'margin-left': 50, 'height': '300px', 'overflowY': 'auto'},
                ),
                             
            ],
        ),
    ],
    className="row twelve columns",
)

def update_graph(
    graph_id,
    graph_title,
    y_train_index,
    y_val_index,
    run_log_json,
    display_mode,
    checklist_smoothing_options,
    slider_smoothing,
    yaxis_title,
):
    """
    :param graph_id: ID for Dash callbacks
    :param graph_title: Displayed on layout
    :param y_train_index: name of column index for y train we want to retrieve
    :param y_val_index: name of column index for y val we want to retrieve
    :param run_log_json: the json file containing the data
    :param display_mode: 'separate' or 'overlap'
    :param checklist_smoothing_options: 'train' or 'val'
    :param slider_smoothing: value between 0 and 1, at interval of 0.05
    :return: dcc Graph object containing the updated figures
    """



@app.callback([
                Output("sim-dur", "children"),
                Output("num-rads-o", "children"),
                Output("arr-rate-1-o", "children"),
                Output("arr-rate-2-o", "children"),
                Output("arr-rate-3-o", "children"),
                Output("proc-rate-1-o", "children"),
                Output("proc-rate-2-o", "children"),
                Output("proc-rate-3-o", "children"),
                Output("targ-rate-1-o", "children"),
                Output("targ-rate-2-o", "children"),
                Output("targ-rate-3-o", "children"),
                Output("results-table", "columns"),
                Output("results-table", "data"),
                Output("results-graph", "figure"),
                Output('idl-plot', 'figure'),   
                Output('busy-plot', 'figure'),      
            ],
            [   Input("button-run-sim", "n_clicks")],
            [
                State("sim-duration", "value"),
                State("num-rads", "value"),
                State("arr-rate-1", "value"),
                State("arr-rate-2", "value"),
                State("arr-rate-3", "value"),
                State("proc-rate-1", "value"),
                State("proc-rate-2", "value"),
                State("proc-rate-3", "value"),
                State("targ-rate-1", "value"),
                State("targ-rate-2", "value"),
                State("targ-rate-3", "value"),
                State("cutoff", "value"),
                State("percent-special", "value"),
                State("verbose-val", "value")
            ],
            )
def update_simulation(n_cl, sim_duration, num_rads, arr_rate_1, arr_rate_2, arr_rate_3, 
                            proc_rate_1, proc_rate_2, proc_rate_3, targ_rate_1, targ_rate_2, targ_rate_3, 
                            cutoff_val, perc_special, verb_val):
    verbose = 0
    if verb_val is not None:
        verbose = 1 
    sim_duration = sim_duration * 60
    arr_rates = [arr_rate_1, arr_rate_2, arr_rate_3]
    proc_rates = [proc_rate_1, proc_rate_2, proc_rate_3]
    targ_rates = [targ_rate_1, targ_rate_2, targ_rate_3]
    sys_state = vru.sim(sim_duration, num_rads, arr_rates, proc_rates, constant_rads=False, cutoff=cutoff_val, verbose=verb_val)
    img_table = sys_state.img_table
    img_table['urgency'] = img_table['urgency'].astype(int).astype(str)
    list_columns = ['img_id','urgency', 'rad_id', 'time_created','time_rad_job_starts', 'time_job_finished', 'wait_time', 'time_w_rad', 'total_time']
    columns = [{"name": i, "id": i} for i in list_columns]
    simulation_params = {'Simulation Duration': sim_duration, 'Num Radiologists': num_rads, 'Avg Time between Images':arr_rates, 'Avg Time to be processed once seen': proc_rates, 'Time Limit': targ_rates}
    #Make figure
    fig = px.scatter(img_table, x="time_created", y="total_time", title="Image Processing Times",
                            labels={
                            "time_created": "Time of Image Creation (minutes into simulation)",
                            "total_time": "Time for Image to be Processed (min)",
                            "urgency" : "Image Urgency",
                            },
                            color="urgency", color_discrete_map={"1": "red", "2": "orange", "3": "green"},
                            hover_name="img_id",
                            log_x=False, size_max=60)
    
    total_per_busy = vru.idle_plots(sys_state.rads)
    image_path = 'figures/idle_plot.jpg'
    #html.Img(src="figures/idle_plot.jpg"),
    #plt.show()
    try:
        idle_img = np.array(Image.open(f"figures/idle_plot.jpg"))
        busy_img = np.array(Image.open(f"figures/busy_plot.jpg"))
    except OSError:
        raise PreventUpdate

    idl_fig = px.imshow(idle_img, color_continuous_scale="gray", title="Busy vs Idle Times for Radiologists", aspect="auto", width=1200, height=900)
    idl_fig.update_layout(coloraxis_showscale=False)
    idl_fig.update_xaxes(showticklabels=False)
    idl_fig.update_yaxes(showticklabels=False)
    idl_fig.update_traces(hovertemplate = None,
                  hoverinfo = "skip")
    
    busy_fig = px.imshow(busy_img, color_continuous_scale="gray", title="Percentage of Time Busy for Radiologists", aspect="auto", width=1000, height=1000)
    busy_fig.update_layout(coloraxis_showscale=False)
    busy_fig.update_xaxes(showticklabels=False)
    busy_fig.update_yaxes(showticklabels=False)
    busy_fig.update_traces(hovertemplate = None,
                  hoverinfo = "skip")

    

    #wt_plot_url = py.plot_mpl(wt_plot, filename="wait_time_plot")
    

    if n_cl in [0, None]:
            raise PreventUpdate
    else:
        return sim_duration, num_rads, u'Urgency 1: {} mins'.format(arr_rate_1), u'Urgency 2: {} mins'.format(arr_rate_2), u'Urgency 3: {} mins'.format(arr_rate_3), u'Urgency 1: {} mins'.format(proc_rate_1), u'Urgency 2: {} mins'.format(proc_rate_2), u'Urgency 3: {} mins'.format(proc_rate_3), u'Urgency 1: {} mins'.format(targ_rate_1), u'Urgency 2: {} mins'.format(targ_rate_2), u'Urgency 3: {} mins'.format(targ_rate_3), columns, img_table.round(2).to_dict('records'), fig, idl_fig, busy_fig #app.get_asset_url(image_path)  

    """""
    simulation_parameters.sim_duration = sim_dur
    simulation_parameters.num_rads = num_rads
    simulation_parameters.arr_rate_1 = arr_rate_1
    simulation_parameters.arr_rate_2 = arr_rate_2
    simulation_parameters.arr_rate_3 = arr_rate_3
    simulation_parameters.proc_rate_1 = proc_rate_1
    simulation_parameters.proc_rate_2 = proc_rate_2
    simulation_parameters.proc_rate_3 = proc_rate_3
    simulation_parameters.cutoff = cutoff
    simulation_parameters.percent_special = per_spec
    simulation_parameters.verbose = verbose_val
    """""




@app.callback(Output("upload-table", "contents"), [Input("demo", "n_clicks")])
def reset_contents(n_clicks):
    if n_clicks:
        return None


@app.callback(
    [Output("demo-explanation", "children"), Output("learn-more-button", "children")],
    [Input("learn-more-button", "n_clicks")],
)
def learn_more(n_clicks):
    if n_clicks is None:
        n_clicks = 0
    if (n_clicks % 2) == 1:
        n_clicks += 1
        return (
            html.Div(
                className="demo_container",
                style={"margin-bottom": "30px"},
                children=[demo_explanation()],
            ),
            "Close",
        )
    n_clicks += 1
    return (html.Div(), "Learn More")


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050, debug=True, use_reloader=False)
