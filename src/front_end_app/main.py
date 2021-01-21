import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output,State,MATCH,ALL,ALLSMALLER
from flask import Flask, Response
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import detect_mask_image
import json
import base64

UPLOAD_DIRECTORY='./uploaded_images'
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


DYNAMIC_CONTROLS = {
    'UI':  dcc.Upload(
        id={
            'type': 'input-data',
            'index': 0
        },
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
    ),
    # 'UV':  dcc.Upload(
    #     id={
    #         'type': 'input-data',
    #         'index': 1
    #     },
    #     children=html.Div([
    #         'Drag and Drop or ',
    #         html.A('Select Files')
    #     ]),
    #     style={
    #         'width': '100%',
    #         'height': '60px',
    #         'lineHeight': '60px',
    #         'borderWidth': '1px',
    #         'borderStyle': 'dashed',
    #         'borderRadius': '5px',
    #         'textAlign': 'center',
    #         'margin': '10px'
    #     },
    # ),
    'LW': html.Button('Start/Stop Web-Cam Feed', id={
            'type': 'input-data',
            'index': 1},
            n_clicks=0)
   }

app.layout = html.Div([
    html.H1('FACE MACK DETECTOR'),
    html.H2('Mode of Use'),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Upload Image', 'value': 'UI'},
            #{'label': 'Upload Video', 'value': 'UV'},
            {'label': 'Live Webcam', 'value': 'LW'}
        ],
        placeholder="Select a mode",
    ),
    html.Div(id='selected-mode'),
    html.Hr(),
    html.H2('Input Data'),
    html.Div(id='input-pane'),
    html.Hr(),
    html.Button('Process', id='process-data',n_clicks=0),
    html.Hr(),
    html.H2('Output Data'),
    html.Div(id='output-pane')
])


@app.callback(
    Output(component_id='selected-mode',component_property= 'children'),
    [Input(component_id='dropdown',component_property= 'value')])
def update_output(value):
    if value=='UI':
        return html.Div([
            DYNAMIC_CONTROLS[value]])
    # elif value=='UV':
    #     return html.Div([
    #         DYNAMIC_CONTROLS[value]])
    elif value == 'LW':
        return (DYNAMIC_CONTROLS[value])

@app.callback(
    Output('input-pane', 'children'),
    Input(component_id='dropdown',component_property= 'value'),
    Input({'type': 'input-data', 'index': ALL}, 'contents'),
    Input({'type': 'input-data', 'index': ALL}, 'n_clicks'),
    State({'type': 'input-data', 'index': ALL}, 'filename'))
def update_output(value,contents,n_clicks,filename):
    ctx=dash.callback_context
    #ctx_msg = json.dumps({
    #    'states': ctx.states,
    #    'triggered': ctx.triggered,
    #    'inputs': ctx.inputs
    #}, indent=2)

    if ctx.triggered:
        usage_mode=ctx.inputs[ "dropdown.value"]

        if usage_mode == 'UI':
            filtered_list=list(filter(lambda x:x["prop_id"]=="{\"index\":0,\"type\":\"input-data\"}.contents",ctx.triggered))
            if filtered_list:
                img_contents=ctx.inputs["{\"index\":0,\"type\":\"input-data\"}.contents"]
                img_filename=ctx.states["{\"index\":0,\"type\":\"input-data\"}.filename"]
                if img_contents:
                    save_file(img_filename,img_contents)
                    return html.Div([html.H5(img_filename),
                                     html.Img(src=img_contents)])
            #elif value == 'UV':
             #   return html.Div([html.H5(filename),
              #                   html.Video(src=contents)])
        elif usage_mode == 'LW':
            filtered_list = list(filter(lambda x: x["prop_id"] == "{\"index\":1,\"type\":\"input-data\"}.n_clicks", ctx.triggered))
            if filtered_list:
                n_clicks=ctx.inputs["{\"index\":1,\"type\":\"input-data\"}.n_clicks"]
                if n_clicks%2 !=0:
                   return html.Div(html.Img(src="/video_feed"))

app.config.suppress_callback_exceptions = True

if __name__ == '__main__':
    app.run_server(debug=True)

