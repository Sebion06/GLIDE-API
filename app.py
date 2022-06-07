from flask import Flask, request, session, jsonify, abort
from flask_restful import Api, Resource, reqparse
from marshmallow import Schema, fields, post_load, validates, ValidationError
from time import process_time
import torch
import glide_api


class ImageSchema(Schema):
    prompt = fields.Str(required=True)
    size = fields.Int(required=False, load_default=64)
    batch_size = fields.Int(required=False, load_default=1)
    guidance_scale = fields.Float(required=False, load_default=3.0)
    upsample_temp = fields.Float(required=False, load_default=0.997)

    @validates('size')
    def validate_size(self, size):
        if size % 64 != 0:
            raise ValidationError("Size should be divisible by 64")

    @validates('upsample_temp')
    def validate_upsample_temp(self, upsample_temp):
        if upsample_temp > 1 or upsample_temp < 0:
            raise ValidationError("Upsample_temp should be between 0 and 1")


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
                        memory_allocated=f"{allocated_memory:.2f} Gb",
                        memory_reserved=f"{reserved_memory:.2f} Gb",
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



base_model, base_options, base_diffusion = glide_api.create_base_model('10')
upsample_model, upsample_options, upsample_diffusion = glide_api.create_upsampler_model('fast27')

app = Flask(__name__, template_folder='template', static_folder='static')
api = Api(app)

api.add_resource(Image, "/image")
api.add_resource(BaseModel, "/basemodel")
api.add_resource(UpModel, "/upmodel")

if __name__ == "__main__":
    app.run(debug=True)