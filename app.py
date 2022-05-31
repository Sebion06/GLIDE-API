import numpy as np
from flask import Flask, request, session, redirect, url_for, jsonify, render_template
import pickle, torch
import script

app = Flask(__name__, template_folder='template', static_folder='static')
app.secret_key = "super secret key"

# batch_size = 1
# guidance_scale = 3.0
# full_batch_size = batch_size * 2

# upsample_temp = 0.997
# base_diffusion_steps = '10'
# umsampler_diffusion_steps = 'fast27'



@app.route('/', methods=['POST', 'GET'])
def home():
    global base_model, base_options, base_diffusion
    global upsample_model, upsample_options, upsample_diffusion

    if request.method == 'POST':
        batch_size = request.form["batch_size"]
        guidance_scale = request.form["guidance_scale"]
        upsample_temp = request.form["upsample_temp"]
        base_diffusion_steps = request.form["base_diffusion_steps"]
        upsampler_diffusion_steps = request.form["upsampler_diffusion_steps"]
        base_model, base_options, base_diffusion = script.create_base_model(base_diffusion_steps)
        upsample_model, upsample_options, upsample_diffusion = script.create_upsampler_model(upsampler_diffusion_steps)

        return redirect(url_for('generate', 
        batch_size=batch_size, guidance_scale=guidance_scale,upsample_temp=upsample_temp))
    return render_template('index.html')


@app.route('/generate',methods=['POST', 'GET'])
def generate():
    global base_model, base_options, base_diffusion
    global upsample_model, upsample_options, upsample_diffusion
    batch_size =  int(session.get('batch_size'))
    guidance_scale =  float(session.get('guidance_scale'))
    upsample_temp =  float(session.get('upsample_temp'))
    full_batch_size = batch_size * 2
    if request.method == 'GET':
        session['batch_size'] = request.args.get('batch_size', None)
        session['guidance_scale'] = request.args.get('guidance_scale', None)
        session['upsample_temp'] = request.args.get('upsample_temp', None)
        

    if request.method == 'POST':
        text_input = request.form["text_input"]

        print(batch_size)


        base_model_kwargs = script.create_base_model_kwargs(base_model, base_options, text_input, batch_size)
        base_model.del_cache()
        base_sample = script.get_base_sample(base_model_kwargs, base_options, base_diffusion, batch_size, full_batch_size)
        base_model.del_cache()
    
        img_path = script.save_images(text_input, base_sample)

        return render_template('generate.html', prediction_image=img_path)
    return render_template('generate.html')



if __name__ == "__main__":
    app.run(debug=True)