from flask import Blueprint, render_template, request
import os
from prediction.prediction import PredictionPipeline
from path.path import PREDICTION_OUTPUT

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
