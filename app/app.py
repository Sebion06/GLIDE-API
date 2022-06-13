import os
import torch
import glide_api
from time import process_time
from flask import Flask, request, session, jsonify, abort, redirect, render_template, url_for
from flask_restful import Api, Resource
from marshmallow import Schema, fields, post_load, validates, ValidationError

base_model, base_options, base_diffusion = glide_api.create_base_model('100')
upsample_model, upsample_options, upsample_diffusion = glide_api.create_upsampler_model('fast27')

app = Flask(__name__, template_folder='template', static_folder='static')
api = Api(app)
app.secret_key = os.urandom(12)

class ImageSchema(Schema):
    prompt = fields.Str(required=True)
    size = fields.Int(required=False, load_default=64)
    batch_size = fields.Int(required=False, load_default=1)
    guidance_scale = fields.Float(required=False, load_default=3.0)
    upsample_temp = fields.Float(required=False, load_default=0.997)

    def is_power_of_two(self, x):
        return (x and (not(x & (x - 1))) )

    @validates('size')
    def validate_size(self, size):
        if size < 64 and not self.is_power_of_two(size):
            raise ValidationError("size should be 64 or bigger and a power of 2")

    @validates('guidance_scale')
    def validate_size(self, guidance_scale):
        if guidance_scale < 1:
            raise ValidationError("guidance_scale should bebigger than 1")

    @validates('upsample_temp')
    def validate_upsample_temp(self, upsample_temp):
        if upsample_temp > 1 or upsample_temp < 0:
            raise ValidationError("upsample_temp should be between 0 and 1")


class ModelSchema(Schema):
    diffusion_steps = fields.Int(required=True)

    @post_load
    def convert_to_string(self, in_data, **kwargs):
        in_data["diffusion_steps"] = str(in_data["diffusion_steps"])
        return in_data
    
    @validates('diffusion_steps')
    def validate_diffusion_steps(self, diffusion_steps):
        if diffusion_steps < 1 or diffusion_steps > 10000:
            raise ValidationError("Value should be between 1 and 10000")


class Image(Resource):
    def get(self):
        image_schema = ImageSchema()
        errors = image_schema.validate(request.args)
        if errors:
            abort(400, str(errors))
        args=image_schema.load(request.args)
        base_size = 64
        t_start = process_time() 
        sample = get_base_sample(args["prompt"], args["batch_size"], args["guidance_scale"])
        while base_size < args["size"]:
            base_size = base_size * 2
            up_sample = get_up_sample(args["prompt"], sample, args["batch_size"], args["upsample_temp"], base_size)
            sample = up_sample
        t_stop = process_time()
        processing_time = t_stop - t_start
        encoded_image = glide_api.encode_image(args["prompt"], sample)
        allocated_memory = torch.cuda.memory_allocated()*pow(10,-9)
        reserved_memory = torch.cuda.memory_reserved()*pow(10,-9)
        return jsonify( data_type="image/png;base64",
                        image_size=args["size"],
                        image=encoded_image,
                        gpu_memory_allocated=f"{allocated_memory:.2f} Gb",
                        gpu_memory_reserved=f"{reserved_memory:.2f} Gb",
                        processing_time = f"{processing_time} s" )


class BaseModel(Resource):
    def post(self):
        global base_model, base_options, base_diffusion
        base_schema = ModelSchema()
        errors = base_schema.validate(request.args)
        if errors:
            abort(400, str(errors))
        args=base_schema.load(request.args)
        base_model, base_options, base_diffusion = glide_api.create_base_model(args["diffusion_steps"])
        return jsonify(base_model_total_parameters = sum(x.numel() for x in base_model.parameters()))


class UpModel(Resource):
    def post(self):
        global upsample_model, upsample_options, upsample_diffusion
        up_schema = ModelSchema()
        errors = up_schema.validate(request.args)
        if errors:
            abort(400, str(errors))
        args=up_schema.load(request.args)
        upsample_model, upsample_options, upsample_diffusion = glide_api.create_upsampler_model(args["diffusion_steps"])
        return jsonify(upsample_model_total_parameters = sum(x.numel() for x in upsample_model.parameters()))


def get_base_sample(text_input, batch_size, guidance_scale):
    global base_model, base_options, base_diffusion
    full_batch_size = batch_size * 2
    base_model_kwargs = glide_api.create_base_model_kwargs(base_model, base_options, text_input, batch_size)
    base_model.del_cache()
    base_sample = glide_api.get_base_sample(base_model_kwargs, base_options, base_diffusion, guidance_scale , batch_size, full_batch_size)
    base_model.del_cache()
    return base_sample


def get_up_sample(text_input, sample, batch_size, upsample_temp, size):
    global upsample_model, upsample_options, upsample_diffusion
    up_model_kwargs = glide_api.create_upsampler_model_kwargs(sample, upsample_model, upsample_options, text_input, batch_size)
    up_sample = glide_api.get_upsampled_sample(up_model_kwargs, upsample_model, upsample_options, upsample_diffusion, batch_size, upsample_temp, size)
    return up_sample


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
            base_sample = get_base_sample(icon_text_input, batch_size, guidance_scale)
            img_path = glide_api.save_images(icon_text_input, base_sample)
            return render_template('generate.html', icon_image=img_path)

        if background_text_input is not None:
            base_sample = get_base_sample(background_text_input, batch_size, guidance_scale)
            up_sample = get_up_sample(background_text_input, base_sample, batch_size, upsample_temp, 256)
            final_sample = get_up_sample(background_text_input, up_sample, batch_size, upsample_temp, 512)
            img_path = glide_api.save_images(background_text_input, final_sample)
            return render_template('generate.html', background_image=img_path)
    return render_template('generate.html')


api.add_resource(Image, "/image")
api.add_resource(BaseModel, "/basemodel")
api.add_resource(UpModel, "/upmodel")


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)