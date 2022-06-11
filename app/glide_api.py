
import os
import base64
import torch as th
from PIL import Image
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)


has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')
model = None
guidance_scale = 3.0

def create_base_model(diffusion_steps):
    global model
    options = model_and_diffusion_defaults()
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = diffusion_steps
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint('base', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))
    return model, options, diffusion;


def create_upsampler_model(diffusion_steps):
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = diffusion_steps
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint('upsample', device))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))
    return model_up, options_up, diffusion_up;


def model_fn(x_t, ts, **kwargs):
    global guidance_scale
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)


def encode_image(prompt, sample):
    file_path = save_images(prompt, sample)
    with open(file_path, "rb") as image_file:
        encoded_img = base64.b64encode(image_file.read())
    if os.path.isfile(file_path) is True:
        os.remove(file_path)
    encoded_string = encoded_img.decode('utf-8')
    return encoded_string


def save_images(prompt, batch: th.Tensor):
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    image = Image.fromarray(reshaped.numpy())
    prompt = prompt.replace(" ", "_")
    count = 1
    if not  os.path.exists("static"):
        os.makedirs("static")
    file_path = f"static/{prompt}.png"
    while os.path.exists(file_path) is True:
        file_path = f"static/{prompt}{count}.png"
        count += 1
    image.save(file_path)
    return file_path


def create_base_model_kwargs(base_model, base_options, prompt, batch_size=1):
    tokens = base_model.tokenizer.encode(prompt)
    tokens, mask = base_model.tokenizer.padded_tokens_and_mask(
        tokens, base_options['text_ctx']
    )
   
    uncond_tokens, uncond_mask = base_model.tokenizer.padded_tokens_and_mask(
        [], base_options['text_ctx']
    )

    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )
    return model_kwargs


def get_base_sample(base_model_kwargs, base_options, base_diffusion, g_scale, batch_size=1, full_batch_size=2):
    global guidance_scale
    guidance_scale = g_scale
    samples = base_diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, base_options["image_size"], base_options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=base_model_kwargs,
        cond_fn=None,
    )[:batch_size]
    return samples


def create_upsampler_model_kwargs(sample, upsample_model, upsample_options, prompt, batch_size=1):
    tokens = upsample_model.tokenizer.encode(prompt)
    tokens, mask = upsample_model.tokenizer.padded_tokens_and_mask(
        tokens, upsample_options['text_ctx']
    )
    model_kwargs = dict(
        low_res=((sample+1)*127.5).round()/127.5 - 1,
        tokens=th.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )
    return model_kwargs


def get_upsampled_sample(up_model_kwargs, upsample_model, upsample_options, upsample_diffusion, batch_size, upsample_temp, image_size):
    upsample_model.del_cache()
    up_shape = (batch_size, 3, image_size, image_size)
    up_samples = upsample_diffusion.ddim_sample_loop(
        upsample_model,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=up_model_kwargs,
        cond_fn=None,
    )[:batch_size]
    return up_samples