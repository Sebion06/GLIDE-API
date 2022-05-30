import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle, torch
import script

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

prompt = "an oil painting of a corgi"
batch_size = 1
guidance_scale = 3.0
full_batch_size = batch_size * 2

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997
base_diffusion_steps = '100'
umsampler_diffusion_steps = 'fast27'


def init():
    base_model, base_options, base_diffusion = script.create_base_model(base_diffusion_steps)
    upsample_model, upsample_options, upsample_diffusion = script.create_upsampler_model(umsampler_diffusion_steps)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate',methods=['POST, GET'])
def generate():

    return render_template('index.html', prediction_image='{}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)