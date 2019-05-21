import os
from flask import Flask, render_template, request, make_response
from functools import wraps, update_wrapper
from PIL import Image
from predictor import Predictor

import numpy as np
import torch

predictor = Predictor()

# Initialize the flask application
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		prediction = predictor.predict(request)
		return prediction
	else:
		return render_template('home.html', prediction=None)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)