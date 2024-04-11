import torch
import math
import k_diffusion.sampling

from k_diffusion.sampling import to_d
from tqdm.auto import trange
from modules import scripts
from modules import sd_samplers_kdiffusion, sd_samplers_common, sd_samplers
from modules.sd_samplers_kdiffusion import KDiffusionSampler

class Smea(scripts.Script):

    def title(self):
        return "Euler Smea Dy sampler"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def __init__(self):
        init()
        return 

def init():
    for i in sd_samplers.all_samplers:
       if "Euler Dy" in i.name:
            return

    samplers_smea = [
        ('Euler Max', sample_euler_max, ['k_euler'], {}),
        ('Euler Dy', sample_euler_dy, ['k_euler'], {}),
        ('Euler Smea', sample_euler_smea, ['k_euler'], {}),
        ('Euler Smea Dy', sample_euler_smea_dy, ['k_euler'], {}),
    ]

    samplers_data_smea = [
        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
        for label, funcname, aliases, options in samplers_smea
        if callable(funcname)
    ]

    sampler_exparams_smea = {
        sample_euler_dy: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_dy: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    }
    sd_samplers_kdiffusion.sampler_extra_params = {**sd_samplers_kdiffusion.sampler_extra_params, **sampler_exparams_smea}
	
    samplers_map_smea = {x.name: x for x in samplers_data_smea}
    sd_samplers_kdiffusion.k_diffusion_samplers_map = {**sd_samplers_kdiffusion.k_diffusion_samplers_map, **samplers_map_smea}

    for i, item in enumerate(sd_samplers.all_samplers):
        if "Euler" in item.name:
            sd_samplers.all_samplers = sd_samplers.all_samplers[:i + 1] + [*samplers_data_smea] + sd_samplers.all_samplers[i + 1:]
            break
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
    sd_samplers.set_samplers()

    return

@torch.no_grad()
def dy_sampling_step(x, model, dt, sigma_hat, **extra_args):
    original_shape = x.shape
    batch_size, m, n = original_shape[0], original_shape[2] // 2, original_shape[3] // 2
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]
    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, 4, m * n, 2, 2)
    c = a_list[:, :, :, 1, 1].view(batch_size, 4, m, n)

    denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **extra_args)
    d = to_d(c, sigma_hat, denoised)
    c = c + d * dt

    d_list = c.view(batch_size, 4, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]
    x = a_list.view(batch_size, 4, m, n, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(batch_size, 4, 2 * m, 2 * n)

    if extra_row or extra_col:
        x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        x_expanded[:, :, :2 * m, :2 * n] = x
        if extra_row:
            x_expanded[:, :, -1:, :2 * n + 1] = extra_row_content
        if extra_col:
            x_expanded[:, :, :2 * m, -1:] = extra_col_content
        if extra_row and extra_col:
            x_expanded[:, :, -1:, -1:] = extra_col_content[:, :, -1:, :]
        x = x_expanded

    return x

@torch.no_grad()
def smea_dy_sampling_step(x, model, dt, sigma_hat, **extra_args):
    m, n = x.shape[2], x.shape[3]
    x = torch.nn.functional.interpolate(input=x, size=None, scale_factor=(1.25, 1.25), mode='nearest-exact', align_corners=None, recompute_scale_factor=None)

    original_shape = x.shape
    batch_size, m, n = original_shape[0], original_shape[2] // 2, original_shape[3] // 2
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]
    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, 4, m * n, 2, 2)
    c = a_list[:, :, :, 1, 1].view(batch_size, 4, m, n)

    denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **extra_args)
    d = to_d(c, sigma_hat, denoised)
    c = c + d * dt

    d_list = c.view(batch_size, 4, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]
    x = a_list.view(batch_size, 4, m, n, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(batch_size, 4, 2 * m, 2 * n)

    if extra_row or extra_col:
        x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        x_expanded[:, :, :2 * m, :2 * n] = x
        if extra_row:
            x_expanded[:, :, -1:, :2 * n + 1] = extra_row_content
        if extra_col:
            x_expanded[:, :, :2 * m, -1:] = extra_col_content
        if extra_row and extra_col:
            x_expanded[:, :, -1:, -1:] = extra_col_content[:, :, -1:, :]
        x = x_expanded

    x = torch.nn.functional.interpolate(input=x, size=(m,n), scale_factor=None, mode='nearest-exact', align_corners=None, recompute_scale_factor=None)
    return x
	
@torch.no_grad()
def smea_sampling_step(x, model, dt, sigma_hat, **extra_args):
    m, n = x.shape[2], x.shape[3]
    x = torch.nn.functional.interpolate(input=x, size=None, scale_factor=(1.25, 1.25), mode='nearest-exact', align_corners=None, recompute_scale_factor=None)
    denoised = model(x, sigma_hat * x.new_ones([x.shape[0]]), **extra_args)
    d = to_d(x, sigma_hat, denoised)
    x = x + d * dt
    x = torch.nn.functional.interpolate(input=x, size=(m,n), scale_factor=None, mode='nearest-exact', align_corners=None, recompute_scale_factor=None)
    return x

@torch.no_grad()
def sample_euler_max(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = max(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x - eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + (math.cos(i + 1)/(i + 1) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_dy(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        # print(i)
        # i第一步为0
        gamma = max(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        # print(sigma_hat)
        dt = sigmas[i + 1] - sigma_hat
        if gamma > 0:
            x = x - eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)	
        if sigmas[i + 1] > 0:
            if i == 1:
                x = dy_sampling_step(x, model, dt, sigma_hat, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        # Euler method
        x = x + d * dt	
    return x

@torch.no_grad()
def sample_euler_smea(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = max(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        dt = sigmas[i + 1] - sigma_hat
        if gamma > 0:
            x = x - eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        # Euler method
        x = x + d * dt
        if sigmas[i + 1] > 0:
            if i == 0:
                x = smea_dy_sampling_step(x, model, dt, sigma_hat, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
    return x

@torch.no_grad()
def sample_euler_smea_dy(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = max(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        dt = sigmas[i + 1] - sigma_hat
        if gamma > 0:
            x = x - eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        # Euler method
        x = x + d * dt
        if sigmas[i + 1] > 0 and i < 3:
            if i % 2 == 0:
                x = dy_sampling_step(x, model, dt, sigma_hat, **extra_args)
            if i % 2 == 1:
                x = smea_sampling_step(x, model, dt, sigma_hat, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
    return x
