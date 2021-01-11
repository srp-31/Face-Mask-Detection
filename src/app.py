import dash
import dash_core_components as dcc
import dash_html_components as html
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


DYNAMIC_CONTROLS = {
    'UI':  dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
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
    'UV':  dcc.Upload(
        id='upload-video',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
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
    'LW':html.Div([
    html.Img(src="/video_feed")
    ])
   }

app.layout = html.Div([
    html.H1('FACE MACK DETECTOR'),
    html.H2('Mode of Use'),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Upload Image', 'value': 'UI'},
            {'label': 'Upload Video', 'value': 'UV'},
            {'label': 'Live Webcam', 'value': 'LW'}
        ],
        placeholder="Select a mode",
    ),
    html.Hr(),
    html.H2('Input pane'),
    html.Div(id='selected-mode'),
    html.Hr(),
    html.H2('Output pane'),
])

@app.callback(
    dash.dependencies.Output(component_id='selected-mode',component_property= 'children'),
    [dash.dependencies.Input(component_id='dropdown',component_property= 'value')])
def update_output(value):

    if value=='UI':
        return html.Div([
            DYNAMIC_CONTROLS[value]])

    elif value=='UV':
        return html.Div([
            DYNAMIC_CONTROLS[value]])
    elif value == 'LW':
        return (DYNAMIC_CONTROLS[value])

if __name__ == '__main__':
    app.run_server(debug=True)
