from flask import Blueprint, render_template, request, send_from_directory, send_file
import os
from prediction.prediction import PredictionPipeline
from path.path import PREDICTION_OUTPUT
from io import BytesIO
from zipfile import ZipFile
from glob import glob

views = Blueprint(__name__, "views")

@views.route("/")
def home():
    return render_template("index.html")

@views.route("/upload", methods=['POST'])
def handle_upload():
    try:
        file = request.files['file']
        filename = os.path.join(os.getcwd(), "data", "prediction_input.xls") 
        file.save(filename)
        return {'status' : 200}
    except:
        return {'status' : 400}
    
@views.route("/predict", methods=['GET'])
def predict_input():
    response = dict()
    status, message = PredictionPipeline().predict()
    if status == True:
        # Send further info
        response['status'] = 200
    else:
        response['status'] = 400
    response['message'] = message
    return response

@views.route("/download", methods=['GET'])
def download():
    print("Calling download")
    return send_from_directory(os.path.join(os.getcwd(), "data"), "prediction_output.csv", as_attachment=True)

@views.route("/logs", methods=['GET'])
def download_logs():
    stream  = BytesIO()
    with ZipFile(stream, 'w') as zf:
        for file in glob(os.path.join(os.getcwd(), "logs", "*.txt")):
            zf.write(file, os.path.basename(file))
    stream.seek(0)

    return send_file (stream, as_attachment=True, download_name='all-logs.zip')