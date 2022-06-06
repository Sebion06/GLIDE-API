import numpy as np
from flask import Flask, request, session, redirect, url_for, jsonify, render_template
import pickle, torch
import glide_api

app = Flask(__name__, template_folder='template', static_folder='static')
app.secret_key = "super secret key"

# batch_size = 1
# guidance_scale = 3.0
# full_batch_size = batch_size * 2

# upsample_temp = 0.997
# base_diffusion_steps = '10'
# umsampler_diffusion_steps = 'fast27'


#TODO: cleanup script
#TODO: add metadata with each request
#TODO: add css
#TODO: maybe add another usecase
#TODO: deploy

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
        base_model, base_options, base_diffusion = glide_api.create_base_model(base_diffusion_steps)
        upsample_model, upsample_options, upsample_diffusion = glide_api.create_upsampler_model(upsampler_diffusion_steps)

        return redirect(url_for('generate', 
        batch_size=batch_size, guidance_scale=guidance_scale,upsample_temp=upsample_temp))
    return render_template('index.html')


@app.route('/generate',methods=['POST', 'GET'])
def generate():
    if request.method == 'GET':
        session['batch_size'] = request.args.get('batch_size', None)
        session['guidance_scale'] = request.args.get('guidance_scale', None)
        session['upsample_temp'] = request.args.get('upsample_temp', None)
        

    batch_size =  int(session.get('batch_size'))
    guidance_scale =  float(session.get('guidance_scale'))
    upsample_temp =  float(session.get('upsample_temp'))


    if request.method == 'POST':
        icon_text_input = request.form.get("icon_text_input")
        background_text_input = request.form.get("background_text_input")
        if icon_text_input is not None:
            base_sample = get_base_sample(icon_text_input, batch_size)
            img_path = glide_api.save_images(icon_text_input, base_sample)
            return render_template('generate.html', icon_image=img_path)

        if background_text_input is not None:
            base_sample = get_base_sample(background_text_input, batch_size)
            up_sample = get_up_sample(background_text_input, base_sample, batch_size, upsample_temp, 256)
            final_sample = get_up_sample(background_text_input, up_sample, batch_size, upsample_temp, 512)
            img_path = glide_api.save_images(background_text_input, final_sample)
            return render_template('generate.html', background_image=img_path)
    return render_template('generate.html')


def get_base_sample(text_input, batch_size):
    global base_model, base_options, base_diffusion
    full_batch_size = batch_size * 2
    base_model_kwargs = glide_api.create_base_model_kwargs(base_model, base_options, text_input, batch_size)
    base_model.del_cache()
    base_sample = glide_api.get_base_sample(base_model_kwargs, base_options, base_diffusion, batch_size, full_batch_size)
    base_model.del_cache()
    return base_sample


def get_up_sample(text_input, sample, batch_size, upsample_temp, size):
    global upsample_model, upsample_options, upsample_diffusion
    up_model_kwargs = glide_api.create_upsampler_model_kwargs(sample, upsample_model, upsample_options, text_input, batch_size)
    up_sample = glide_api.get_upsampled_sample(up_model_kwargs, upsample_model, upsample_options, upsample_diffusion, batch_size, upsample_temp, size)
    return up_sample


if __name__ == "__main__":
    app.run(debug=True)