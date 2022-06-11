import torch as th
from glide_api import create_base_model, create_upsampler_model


has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')
guidance_scale = 3.0
default_base_diff_steps = '100'
default_up_diff_steps = '27'

create_base_model(default_base_diff_steps)
create_upsampler_model(default_up_diff_steps)