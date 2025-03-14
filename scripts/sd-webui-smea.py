import torch
import math
import k_diffusion.sampling

from k_diffusion.sampling import to_d, BrownianTreeNoiseSampler
from tqdm.auto import trange
from modules import scripts
from modules import sd_samplers_kdiffusion, sd_samplers_common, sd_samplers
from modules.sd_samplers_kdiffusion import KDiffusionSampler

class _Rescaler:
    def __init__(self, model, x, mode, **extra_args):
        self.model = model
        self.x = x
        self.mode = mode
        self.extra_args = extra_args
        self.init_latent, self.mask, self.nmask = model.init_latent, model.mask, model.nmask

    def __enter__(self):
        if self.init_latent is not None:
            self.model.init_latent = torch.nn.functional.interpolate(input=self.init_latent, size=self.x.shape[2:4], mode=self.mode)
        if self.mask is not None:
            self.model.mask = torch.nn.functional.interpolate(input=self.mask.unsqueeze(0), size=self.x.shape[2:4], mode=self.mode).squeeze(0)
        if self.nmask is not None:
            self.model.nmask = torch.nn.functional.interpolate(input=self.nmask.unsqueeze(0), size=self.x.shape[2:4], mode=self.mode).squeeze(0)
        return self

    def __exit__(self, type, value, traceback):
        del self.model.init_latent, self.model.mask, self.model.nmask
        self.model.init_latent, self.model.mask, self.model.nmask = self.init_latent, self.mask, self.nmask

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
       if "Euler Max" in i.name:
            return

    samplers_smea = [
        ('Euler Max', sample_euler_max, ['k_euler'], {}),
        ('Euler Max1b', sample_euler_max1b, ['k_euler'], {}),
        ('Euler Max1c', sample_euler_max1c, ['k_euler'], {}),
        ('Euler Max1d', sample_euler_max1d, ['k_euler'], {}),
        ('Euler Max2', sample_euler_max2, ['k_euler'], {}),
        ('Euler Max2b', sample_euler_max2b, ['k_euler'], {}),
        ('Euler Max2c', sample_euler_max2c, ['k_euler'], {}),
        ('Euler Max2d', sample_euler_max2d, ['k_euler'], {}),
        ('Euler Max3', sample_euler_max3, ['k_euler'], {}),
        ('Euler Max3b', sample_euler_max3b, ['k_euler'], {}),
        ('Euler Max3c', sample_euler_max3c, ['k_euler'], {}),
        ('Euler Max4', sample_euler_max4, ['k_euler'], {}),
        ('Euler Max4b', sample_euler_max4b, ['k_euler'], {}),
        ('Euler Max4c', sample_euler_max4c, ['k_euler'], {}),
        ('Euler Max4d', sample_euler_max4d, ['k_euler'], {}),
        ('Euler Max4e', sample_euler_max4e, ['k_euler'], {}),
        ('Euler Max4f', sample_euler_max4f, ['k_euler'], {}),
        ('Euler Dy', sample_euler_dy, ['k_euler'], {}),
        ('Euler Smea', sample_euler_smea, ['k_euler'], {}),
        ('Euler Smea Dy', sample_euler_smea_dy, ['k_euler'], {}),
        ('Euler Smea Max', sample_euler_smea_max, ['k_euler'], {}),		
        ('Euler Smea Max s', sample_euler_smea_max_s, ['k_euler'], {}),
        ('Euler Smea dyn a', sample_euler_smea_dyn_a, ['k_euler'], {}),
        ('Euler Smea dyn b', sample_euler_smea_dyn_b, ['k_euler'], {}),
        ('Euler Smea dyn c', sample_euler_smea_dyn_c, ['k_euler'], {}),
        ('Euler Smea ma', sample_euler_smea_multi_a, ['k_euler'], {}),
        ('Euler Smea mb', sample_euler_smea_multi_b, ['k_euler'], {}),
        ('Euler Smea mc', sample_euler_smea_multi_c, ['k_euler'], {}),
        ('Euler Smea md', sample_euler_smea_multi_d, ['k_euler'], {}),
        ('Euler Smea mas', sample_euler_smea_multi_as, ['k_euler'], {}),
        ('Euler Smea mbs', sample_euler_smea_multi_bs, ['k_euler'], {}),
        ('Euler Smea mcs', sample_euler_smea_multi_cs, ['k_euler'], {}),
        ('Euler Smea mds', sample_euler_smea_multi_ds, ['k_euler'], {}),
        ('Euler Smea mbs2', sample_euler_smea_multi_bs2, ['k_euler'], {}),
        ('Euler Smea mds2', sample_euler_smea_multi_ds2, ['k_euler'], {}),
        ('Euler Smea mds2 max', sample_euler_smea_multi_ds2_m, ['k_euler'], {}),
        ('Euler Smea mds2 s max', sample_euler_smea_multi_ds2_s_m, ['k_euler'], {}),
        ('Euler Smea mbs2 s', sample_euler_smea_multi_bs2_s, ['k_euler'], {}),
        ('Euler Smea mds2 s', sample_euler_smea_multi_ds2_s, ['k_euler'], {}),
        ('Euler h max', sample_euler_h_m, ['k_euler'], {"brownian_noise": True}),	
        ('Euler h max b', sample_euler_h_m_b, ['k_euler'], {"brownian_noise": True}),
        ('Euler h max c', sample_euler_h_m_c, ['k_euler'], {"brownian_noise": True}),
        ('Euler h max d', sample_euler_h_m_d, ['k_euler'], {"brownian_noise": True}),
        ('Euler h max e', sample_euler_h_m_e, ['k_euler'], {"brownian_noise": True}),
        ('Euler h max f', sample_euler_h_m_f, ['k_euler'], {"brownian_noise": True}),
        ('Euler h max g', sample_euler_h_m_g, ['k_euler'], {"brownian_noise": True}),
        ('Euler h max b c', sample_euler_h_m_b_c, ['k_euler'], {"brownian_noise": True}),
        ('Euler h max b c CFG++', sample_euler_h_m_b_c_pp, ['k_euler'], {"brownian_noise": True, "cfgpp": True}),
        ('Euler Dy koishi-star', sample_euler_dy_og, ['k_euler'], {}),
        ('Euler Smea Dy koishi-star', sample_euler_smea_dy_og, ['k_euler'], {}),
        ('TCD Euler a', sample_tcd_euler_a, ['tcd_euler_a'], {}),
        ('TCD', sample_tcd, ['tcd'], {}),
    ]

    samplers_data_smea = [
        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
        for label, funcname, aliases, options in samplers_smea
        if callable(funcname)
    ]

    sampler_exparams_smea = {
        sample_euler_max: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max1b: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max1c: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max1d: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max2: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max2b: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max2c: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max2d: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max3: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max3b: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max3c: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max4: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max4b: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max4c: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max4d: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max4e: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_max4f: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_dy: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_dy: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_max: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_max_s: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_dyn_a: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_dyn_b: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_dyn_c: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_a: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_b: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_c: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_d: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_as: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_bs: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_cs: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_ds: ['s_churn', 's_tmin', 's_tmax', 's_noise'],            
        sample_euler_smea_multi_bs2: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_ds2: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_ds2_m: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_ds2_s_m: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_bs2_s: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_multi_ds2_s: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_h_m: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_h_m_b: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_h_m_c: ['s_churn', 's_tmin', 's_tmax', 's_noise'],   
        sample_euler_h_m_d: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_h_m_e: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_h_m_f: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_h_m_g: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_h_m_b_c: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_h_m_b_c_pp: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_dy_og: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
        sample_euler_smea_dy_og: ['s_churn', 's_tmin', 's_tmax', 's_noise'],
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

def default_noise_sampler(x):
    return lambda sigma, sigma_next: k_diffusion.sampling.torch.randn_like(x)

@torch.no_grad()
def dy_sampling_step(x, model, dt, sigma_hat, **extra_args):
    original_shape = x.shape
    batch_size, channels, m, n = original_shape[0], original_shape[1], original_shape[2] // 2, original_shape[3] // 2
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]
    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, channels, m * n, 2, 2)
    c = a_list[:, :, :, 1, 1].view(batch_size, channels, m, n)

    with _Rescaler(model, c, 'nearest-exact', **extra_args) as rescaler:
        denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **rescaler.extra_args)
    d = to_d(c, sigma_hat, denoised)
    c = c + d * dt

    d_list = c.view(batch_size, channels, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]
    x = a_list.view(batch_size, channels, m, n, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(batch_size, channels, 2 * m, 2 * n)

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
def smea_sampling_step(x, model, dt, sigma_hat, **extra_args):
    m, n = x.shape[2], x.shape[3]
    x = torch.nn.functional.interpolate(input=x, size=None, scale_factor=(1.25, 1.25), mode='nearest-exact', align_corners=None, recompute_scale_factor=None)
    with _Rescaler(model, x, 'nearest-exact', **extra_args) as rescaler:
        denoised = model(x, sigma_hat * x.new_ones([x.shape[0]]), **rescaler.extra_args)
    d = to_d(x, sigma_hat, denoised)
    x = x + d * dt
    x = torch.nn.functional.interpolate(input=x, size=(m,n), scale_factor=None, mode='nearest-exact', align_corners=None, recompute_scale_factor=None)
    return x

@torch.no_grad()
def smea_sampling_step_denoised(x, model, sigma_hat, scale=1.25, smooth=False, **extra_args):
    m, n = x.shape[2], x.shape[3]
    filter = 'nearest-exact' if not smooth else 'bilinear'
    x = torch.nn.functional.interpolate(input=x, scale_factor=(scale, scale), mode=filter)
    with _Rescaler(model, x, filter, **extra_args) as rescaler:
        denoised = model(x, sigma_hat * x.new_ones([x.shape[0]]), **rescaler.extra_args)
    x = denoised
    x = torch.nn.functional.interpolate(input=x, size=(m,n), mode='nearest-exact')
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
def sample_euler_max1b(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(1.05 * i + 1)/(1.1 * i + 1.5) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_max1c(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(1.05 * i + 1.1)/(1.25 * i + 1.5) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_max1d(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(math.pi * 0.333 * i + 0.9)/(0.5 * i + 1.5) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_max2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(math.pi * 0.333 * i - 0.1)/(0.5 * i + 1.5) + 1) * d * dt
    return x
	
@torch.no_grad()
def sample_euler_max2b(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(math.pi * 0.5 * i - 0.0)/(0.5 * i + 1.5) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_max2c(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(math.pi * 0.5 * i)/(i + 2) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_max2d(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(math.pi * 0.5 * i)/(0.75 * i + 1.75) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_max3b(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(2 * i + 0.5)/(2 * i + 1.5) + 1) * d * dt
    return x
	
@torch.no_grad()
def sample_euler_max3c(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(2 * i + 0.5)/(1.5 * i + 2.7) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_max3(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(2 * i + 1)/(2 * i + 1) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_max4b(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(math.pi * i - 0.1)/(2 * i + 2) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_max4c(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(math.pi * i - 0.1)/(2 * i + 1.5) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_max4d(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(math.pi * i - 0.1)/(i + 1.5) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_max4e(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(math.pi * i - 0.1)/(i + 1) + 1) * d * dt
    return x

@torch.no_grad()
def sample_euler_max4f(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(math.pi * i - 0.1)/(i + 2) + 1) * d * dt
    return x
	
@torch.no_grad()
def sample_euler_max4(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + (math.cos(math.pi * i - 0.1)/(math.pi * 0.5 * i + math.pi * 0.5) + 1) * d * dt
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
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.334 - len(sigmas) * 0.334 % 2 and i % 2 == 0:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigmas[i + 1] - sigmas[i]
            x_2 = x + d * dt_1
            x_temp = dy_sampling_step(x_2, model, dt_2, sigma_mid, **extra_args)
            x = x_temp - d * dt_1
        # Euler method
        x = x + d * dt	
    return x

@torch.no_grad()
def sample_euler_smea_dyn_a(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.334 - len(sigmas) * 0.334 % 2:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            #scale = (sigma_mid / sigmas[0]) * 0.25
            scale = ((len(sigmas) - i) / len(sigmas)) ** 2 * 0.15
            #scale = scale.item()
            if i % 2 == 0:
                denoised_2 = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 + scale, **extra_args)
                #denoised_2 = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 + sigma_mid.item() * 0.01, **extra_args)
            else:
                denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_smea_dyn_b(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and (i < len(sigmas) * 0.334 - len(sigmas) * 0.334 % 3 or i < 3):
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
	    #scale = (sigma_mid / sigmas[0]) * 0.25
            scale = ((len(sigmas) - i) / len(sigmas)) ** 2 * 0.2
            #scale = scale.item()
            if i % 4 == 0:
                denoised_2 = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 - scale, **extra_args)
	        #denoised_2 = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 - sigma_mid.item() * 0.01, **extra_args)
            elif i % 4 == 2:
                denoised_2 = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 + scale, **extra_args)
		#denoised_2 = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 + sigma_mid.item() * 0.01, **extra_args)
            else:
                denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_smea_dyn_c(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.334 - len(sigmas) * 0.334 % 2:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            #scale = (sigma_mid / sigmas[0]) * 0.25
            scale = ((len(sigmas) - i) / len(sigmas)) ** 2 * 0.25
            #scale = scale.item()
            if i % 2 == 0:
                denoised_2 = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 - scale, **extra_args)
                #denoised_2 = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 + sigma_mid.item() * 0.01, **extra_args)
            else:
                denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
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
        if gamma > 0:
            x = x - eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.334 - len(sigmas) * 0.334 % 2 and i % 2 == 0:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigmas[i + 1] - sigmas[i]
            #print(dt_1, "#", dt_2, "#", dt_3, "#", dt_4)
            x_2 = x + d * dt_1
            x_temp = smea_sampling_step(x, model, dt_2, sigma_mid, **extra_args)
            x = x_temp - d * dt_1
    return x

@torch.no_grad()
def sample_euler_smea_dy(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
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
        x = x + d * dt
        if sigmas[i + 1] > 0 and (i < len(sigmas) * 0.334 - len(sigmas) * 0.334 % 2 or i < 3) and i % 3 != 2:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigmas[i + 1] - sigmas[i]
            #print(dt_1, "#", dt_2, "#", dt_3, "#", dt_4)
            x_2 = x + d * dt_1
            if i % 3 == 1:
                x_temp = dy_sampling_step(x, model, dt_2, sigma_mid, **extra_args)
            elif i % 3 == 0:
                x_temp = smea_sampling_step(x, model, dt_2, sigma_mid, **extra_args)
            x = x_temp - d * dt_1
    return x

@torch.no_grad()
def sample_euler_smea_multi_d(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.334 + 2 and i % 2 == 0:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            scale = ((len(sigmas) - i) / len(sigmas)) ** 2
            if i == 0:
                denoised_2a = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 - scale * 0.15, **extra_args)
                denoised_2c = model(x_2, sigma_mid * s_in, **extra_args)
                denoised_2 = (denoised_2a + denoised_2c) / 2
            elif i < len(sigmas) * 0.334:
                denoised_2a = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 - scale * 0.25, **extra_args)
                denoised_2b = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 + scale * 0.15, **extra_args)
                denoised_2c = model(x_2, sigma_mid * s_in, **extra_args)
                denoised_2 = (denoised_2a + denoised_2b + denoised_2c) / 3
            else:
                denoised_2b = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 + scale * 0.03, True, **extra_args)
                denoised_2c = model(x_2, sigma_mid * s_in, **extra_args)
                denoised_2 = (denoised_2b + denoised_2c) / 2
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_smea_multi_b(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.167:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            scale = ((len(sigmas) - i) / len(sigmas)) ** 2
            denoised_2a = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 - scale * 0.25, **extra_args)
            denoised_2b = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 + scale * 0.15, **extra_args)
            denoised_2c = model(x_2, sigma_mid * s_in, **extra_args)
            denoised_2 = (denoised_2a + denoised_2b + denoised_2c) / 3
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
    return x
	
@torch.no_grad()
def sample_euler_smea_multi_c(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.167:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            scale = ((len(sigmas) - i) / len(sigmas)) ** 2
            denoised_2a = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 - scale * 0.25, **extra_args)
            denoised_2c = model(x_2, sigma_mid * s_in, **extra_args)
            denoised_2 = (denoised_2a + denoised_2c) / 2
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_smea_multi_a(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.167:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            scale = ((len(sigmas) - i) / len(sigmas)) ** 2
            denoised_2b = smea_sampling_step_denoised(x_2, model, sigma_mid, 1 + scale * 0.15, **extra_args)
            denoised_2c = model(x_2, sigma_mid * s_in, **extra_args)
            denoised_2 = (denoised_2b + denoised_2c) / 2
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
    return x

	
@torch.no_grad()
def sample_euler_smea_multi_ds(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.167 + 1: # and i % 2 == 0:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            scale = ((len(sigmas) - i) / len(sigmas)) ** 2
            if i == 0:
                sa = 1 - scale * 0.15
                sb = 1 + scale * 0.09	
                denoised_2a = smea_sampling_step_denoised(x_2, model, sigma_mid, sa, **extra_args)
                denoised_2b = smea_sampling_step_denoised(x_2, model, sigma_mid, sb, **extra_args)
                denoised_2 = (denoised_2a * (sa ** 2) * 0.625 + denoised_2b * (sb ** 2) * 0.375) / (0.97**2)
            elif i < len(sigmas) * 0.167:
                sa = 1 - scale * 0.25
                sb = 1 + scale * 0.15
                denoised_2a = smea_sampling_step_denoised(x_2, model, sigma_mid, sa, **extra_args)
                denoised_2b = smea_sampling_step_denoised(x_2, model, sigma_mid, sb , **extra_args)
                denoised_2 = (denoised_2a * (sa ** 2) * 0.625 + denoised_2b * (sb ** 2) * 0.375) / (0.95**2)
            else:
                sb = 1 + scale * 0.06
                sc = 1 - scale * 0.1
                denoised_2b = smea_sampling_step_denoised(x_2, model, sigma_mid, sb, True, **extra_args)
                denoised_2c = smea_sampling_step_denoised(x_2, model, sigma_mid, sc, **extra_args)
                denoised_2 = (denoised_2b * (sb ** 2) * 0.375 + denoised_2c * (sc ** 2) * 0.625) / (0.98**2)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
    return x	

@torch.no_grad()
def sample_euler_smea_multi_ds2_s(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    sample = sample_euler_smea_multi_ds2(model, x, sigmas, extra_args, callback, disable, s_churn, s_tmin, s_tmax, s_noise, smooth=True)
    return sample

@torch.no_grad()
def sample_euler_smea_multi_ds2_s_m(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    sample = sample_euler_smea_multi_ds2_m(model, x, sigmas, extra_args, callback, disable, s_churn, s_tmin, s_tmax, s_noise, smooth=True)
    return sample

@torch.no_grad()
def sample_euler_smea_multi_ds2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., smooth=False):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.167 + 1: # and i % 2 == 0:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            scale = (sigmas[i] / sigmas[0]) ** 2
            scale = scale.item()
            if i == 0:
                sa = 1 - scale * 0.15
                sb = 1 + scale * 0.09	
                sigA = sigma_mid / (sa ** 2)
                sigB = sigma_mid / (sb ** 2)
                denoised_2a = smea_sampling_step_denoised(x_2, model, sigA, sa, smooth, **extra_args)
                denoised_2b = smea_sampling_step_denoised(x_2, model, sigB, sb, smooth, **extra_args)
                denoised_2 = (denoised_2a * (sa ** 2) * 0.5 * sb ** 2 + denoised_2b * (sb ** 2) * 0.5 * sa ** 2) #/ (0.97**2) # 1 - (sa * sb ) / 2 + 1
                d_2 = to_d(x_2, sigA * 0.5 * sb ** 2 + sigB * 0.5 * sa ** 2, denoised_2)
            elif i < len(sigmas) * 0.167:
                sa = 1 - scale * 0.25
                sb = 1 + scale * 0.15
                sigA = sigma_mid / (sa ** 2)
                sigB = sigma_mid / (sb ** 2)
                denoised_2a = smea_sampling_step_denoised(x_2, model, sigA, sa, smooth, **extra_args)
                denoised_2b = smea_sampling_step_denoised(x_2, model, sigB, sb, smooth, **extra_args)
                denoised_2 = (denoised_2a * (sa ** 2) * 0.5 * sb ** 2 + denoised_2b * (sb ** 2) * 0.5 * sa ** 2) #/ (0.95**2)
                d_2 = to_d(x_2, sigA * 0.5 * sb ** 2 + sigB * 0.5 * sa ** 2, denoised_2)
            else:
                sb = 1 + scale * 0.06
                sc = 1 - scale * 0.1
                sigB = sigma_mid / (sb ** 2)
                sigC = sigma_mid / (sc ** 2)
                denoised_2b = smea_sampling_step_denoised(x_2, model, sigB, sb, smooth, **extra_args)
                denoised_2c = smea_sampling_step_denoised(x_2, model, sigC, sc, smooth, **extra_args)
                denoised_2 = (denoised_2b * (sb ** 2) * 0.5 * sc ** 2 + denoised_2c * (sc ** 2) * 0.5 * sb ** 2) #/ (0.98**2)
                d_2 = to_d(x_2, sigB * 0.5 * sc ** 2 + sigC * 0.5 * sb ** 2, denoised_2)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_smea_multi_ds2_m(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., smooth=False):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.167 + 1: # and i % 2 == 0:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            scale = (sigmas[i] / sigmas[0]) ** 2
            #scale = dt_1 ** 2 * 0.01
            scale = scale.item()
            if i == 0:
                sa = 1 - scale * 0.15 #15
                sb = 1 + scale * 0.09 #09	
                sigA = sigma_mid / (sa ** 2)
                sigB = sigma_mid / (sb ** 2)
                #delta = sa * sb
                denoised_2a = smea_sampling_step_denoised(x_2, model, sigA, sa, smooth, **extra_args)
                denoised_2b = smea_sampling_step_denoised(x_2, model, sigB, sb, smooth, **extra_args)
                denoised_2 = (denoised_2a * (sa ** 2) * 0.5 * sb ** 2 + denoised_2b * (sb ** 2) * 0.5 * sa ** 2) #/ (0.97**2) # 1 - (sa * sb ) / 2 + 1
                d_2 = to_d(x_2, sigA * 0.5 * sb ** 2 + sigB * 0.5 * sa ** 2, denoised_2)
            elif i < len(sigmas) * 0.167:
                sa = 1 - scale * 0.25 #25
                sb = 1 + scale * 0.15 #15
                sigA = sigma_mid / (sa ** 2)
                sigB = sigma_mid / (sb ** 2)
                #delta = sa * sb
                denoised_2a = smea_sampling_step_denoised(x_2, model, sigA, sa, smooth, **extra_args)
                denoised_2b = smea_sampling_step_denoised(x_2, model, sigB, sb, smooth, **extra_args)
                denoised_2 = (denoised_2a * (sa ** 2) * 0.5 * sb ** 2 + denoised_2b * (sb ** 2) * 0.5 * sa ** 2) #/ (0.95**2)
                d_2 = to_d(x_2, sigA * 0.5 * sb ** 2 + sigB * 0.5 * sa ** 2, denoised_2)
            else:
                sb = 1 + scale * 0.06
                sc = 1 - scale * 0.1
                sigB = sigma_mid / (sb ** 2)
                sigC = sigma_mid / (sc ** 2)
                #delta = sb * sc
                denoised_2b = smea_sampling_step_denoised(x_2, model, sigB, sb, smooth, **extra_args)
                denoised_2c = smea_sampling_step_denoised(x_2, model, sigC, sc, smooth, **extra_args)
                denoised_2 = (denoised_2b * (sb ** 2) * 0.5 * sc ** 2+ denoised_2c * (sc ** 2) * 0.5 * sb ** 2) #/ (0.98**2)
                d_2 = to_d(x_2, sigB * 0.5 * sc ** 2 + sigC * 0.5 * sb ** 2, denoised_2)
            x = x + (math.cos(1.05 * i + 1.1)/(1.25 * i + 1.5) + 1) * d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + (math.cos(1.05 * i + 1.1)/(1.25 * i + 1.5) + 1) * d * dt
    return x	

@torch.no_grad()
def sample_euler_h_m(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        wave = math.cos(math.pi * 0.5 * i)/(0.5 * i + 1.5) + 1
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        s_tmin, s_tmax = sigma_min if s_tmin == 0. else s_tmin, sigma_max if s_tmax == float('inf') else s_tmax
        gamma = min((2 ** 0.5 - 1) - wave * ((2 ** 0.5 - 1) + s_churn) / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.BrownianTreeNoiseSampler(x, s_tmin, s_tmax, 0) if noise_sampler == None else noise_sampler
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x - eps(sigmas[i], sigmas[i + 1]) * s_noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * (gamma + 1), denoised)
            d_prime = d * 0.5 + d_2 * 0.5
            x = x + d_prime * dt
        else:
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_h_m_b(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        wave = math.cos(math.pi * 0.5 * i)/(0.5 * i + 1.5) + 1
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        s_tmin, s_tmax = sigma_min if s_tmin == 0. else s_tmin, sigma_max if s_tmax == float('inf') else s_tmax
        gamma = min(wave * ((2 ** 0.5 - 1) + s_churn) / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.BrownianTreeNoiseSampler(x, s_tmin, s_tmax, 0) if noise_sampler is None else noise_sampler
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps(sigmas[i], sigmas[i + 1]) * s_noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * (gamma + 1), denoised)
            d_prime = d * 0.5 + d_2 * 0.5
            x = x + d_prime * dt
        else:
            # Euler method
            x = x + d * dt
    return x	

@torch.no_grad()
def sample_euler_h_m_c(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        wave = math.cos(math.pi * 0.5 * i)/(0.5 * i + 1.5) + 1
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        s_tmin, s_tmax = sigma_min if s_tmin == 0. else s_tmin, sigma_max if s_tmax == float('inf') else s_tmax
        gamma = max((2 ** 0.5 - 1) + wave * ((2 ** 0.5 - 1) + s_churn) / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.BrownianTreeNoiseSampler(x, s_tmin, s_tmax, 0) if noise_sampler is None else noise_sampler
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps(sigmas[i], sigmas[i + 1]) * s_noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * (gamma + 1), denoised)
            d_prime = d * 0.5 + d_2 * 0.5
            x = x + d_prime * dt
        else:
            # Euler method
            x = x + d * dt
    return x	

@torch.no_grad()
def sample_euler_h_m_d(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        wave = math.cos(math.pi * 0.5 * i)/(0.5 * i + 1.5) + 1
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        s_tmin, s_tmax = sigma_min if s_tmin == 0. else s_tmin, sigma_max if s_tmax == float('inf') else s_tmax
        gamma = min((2 ** 0.5 - 1) - wave * ((2 ** 0.5 - 1) + s_churn) / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.BrownianTreeNoiseSampler(x, s_tmin, s_tmax, 0) if noise_sampler is None else noise_sampler
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps(sigmas[i], sigmas[i + 1]) * s_noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * (gamma + 1), denoised)
            d_prime = d * 0.5 + d_2 * 0.5
            x = x + d_prime * dt
        else:
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_h_m_e(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        wave = math.cos(math.pi * 0.5 * i)/(0.5 * i + 1.5) + 1
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        s_tmin, s_tmax = sigma_min if s_tmin == 0. else s_tmin, sigma_max if s_tmax == float('inf') else s_tmax
        gamma = max((2 ** 0.5 - 1) + wave * ((2 ** 0.5 - 1) + s_churn) / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.BrownianTreeNoiseSampler(x, s_tmin, s_tmax, 0) if noise_sampler is None else noise_sampler
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x - eps(sigmas[i], sigmas[i + 1]) * s_noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * (gamma + 1), denoised)
            d_prime = d * 0.5 + d_2 * 0.5
            x = x + d_prime * dt
        else:
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_h_m_f(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        wave = math.cos(math.pi * 0.5 * i)/(0.5 * i + 1.5) + 1
        wave_max = math.cos(0)/1.5 + 1
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        s_tmin, s_tmax = sigma_min if s_tmin == 0. else s_tmin, sigma_max if s_tmax == float('inf') else s_tmax
        gamma = min((wave_max - wave) * ((2 ** 0.5 - 1) + s_churn) / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.BrownianTreeNoiseSampler(x, s_tmin, s_tmax, 0) if noise_sampler is None else noise_sampler
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x - eps(sigmas[i], sigmas[i + 1]) * s_noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * (gamma + 1), denoised)
            d_prime = d * 0.5 + d_2 * 0.5
            x = x + d_prime * dt
        else:
            # Euler method
            x = x + d * dt
    return x	

@torch.no_grad()
def sample_euler_h_m_g(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        wave = math.cos(math.pi * 0.5 * i)/(0.5 * i + 1.5) + 1
        wave_max = math.cos(0)/1.5 + 1
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        s_tmin, s_tmax = sigma_min if s_tmin == 0. else s_tmin, sigma_max if s_tmax == float('inf') else s_tmax
        gamma = min((wave_max - wave) * ((2 ** 0.5 - 1) + s_churn) / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.BrownianTreeNoiseSampler(x, s_tmin, s_tmax, 0) if noise_sampler is None else noise_sampler
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps(sigmas[i], sigmas[i + 1]) * s_noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * (gamma + 1), denoised)
            d_prime = d * 0.5 + d_2 * 0.5
            x = x + d_prime * dt
        else:
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_h_m_b_c(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        wave = math.cos(math.pi * 0.5 * i)/(0.5 * i + 1.5) + 1
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        s_tmin, s_tmax = sigma_min if s_tmin == 0. else s_tmin, sigma_max if s_tmax == float('inf') else s_tmax
        gamma = min(wave * ((2 ** 0.5 - 1) + s_churn) / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.BrownianTreeNoiseSampler(x, s_tmin, s_tmax, 0) if noise_sampler is None else noise_sampler
        gammaup = gamma + 1
        sigma_hat = sigmas[i] * gammaup
        if gamma > 0:	
            x = x + eps(sigmas[i], sigmas[i + 1]) * s_noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        last_noise_uncond = model.last_noise_uncond
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if i == 0:
            x = x + d * dt
        elif i <= len(sigmas) - 4:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * gammaup, denoised)
            x_3 = x_2 + d_2 * dt
            d_3 = to_d(x_3, sigmas[i + 2] * gammaup, denoised)
            d_prime = d * 0.5 + d_2 * 0.375 + d_3 * 0.125		
            x = x + d_prime * dt
        elif sigmas[i + 1] > 0:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * gammaup, denoised)
            d_prime = d * 0.5 + d_2 * 0.5		
            x = x + d_prime * dt
        else:
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_h_m_b_c_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        wave = math.cos(math.pi * 0.5 * i)/(0.5 * i + 1.5) + 1
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        s_tmin, s_tmax = sigma_min if s_tmin == 0. else s_tmin, sigma_max if s_tmax == float('inf') else s_tmax
        gamma = min(wave * ((2 ** 0.5 - 1) + s_churn) / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.BrownianTreeNoiseSampler(x, s_tmin, s_tmax, 0) if noise_sampler is None else noise_sampler
        gammaup = gamma + 1
        sigma_hat = sigmas[i] * gammaup
        if gamma > 0:	
            x = x + eps(sigmas[i], sigmas[i + 1]) * s_noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        last_noise_uncond = model.last_noise_uncond
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if i == 0:
            x = x + d * dt
        elif i <= len(sigmas) - 4:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * gammaup, denoised)
            x_3 = x_2 + d_2 * dt
            d_3 = to_d(x_3, sigmas[i + 2] * gammaup, last_noise_uncond)
            d_prime = d * 0.5 + d_2 * 0.375 + d_3 * 0.125		
            x = x + d_prime * dt
        elif sigmas[i + 1] > 0:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1] * gammaup, denoised)
            d_prime = d * 0.5 + d_2 * 0.5		
            x = x + d_prime * dt
        else:
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_smea_max(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., smooth=False):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        sa = math.cos(i + 1)/(1.5 * i + 1.75) + 1
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.167 + 1: # and i % 2 == 0:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            sigA = sigma_mid / (sa ** 2)
            sigB = sigma_mid
            denoised_2a = smea_sampling_step_denoised(x_2, model, sigA, sa, smooth, **extra_args)
            denoised_2b = model(x_2, sigma_mid * s_in, **extra_args)
            denoised_2 = (denoised_2a * 0.5 * (sa ** 2) + denoised_2b * 0.5 / (sa ** 2))
            d_2 = to_d(x_2, sigA * 0.5 * (sa ** 2) + sigB * 0.5 / (sa ** 2), denoised_2)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + sa * d * dt
    return x	


@torch.no_grad()
def sample_euler_smea_max_s(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    sample = sample_euler_smea_max(model, x, sigmas, extra_args, callback, disable, s_churn, s_tmin, s_tmax, s_noise, smooth=True)
    return sample

@torch.no_grad()
def sample_euler_smea_multi_bs(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.167:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            scale = ((len(sigmas) - i) / len(sigmas)) ** 2
            sa = 1 - scale * 0.25
            sb = 1 + scale * 0.15
            denoised_2a = smea_sampling_step_denoised(x_2, model, sigma_mid, sa, **extra_args) 
            denoised_2b = smea_sampling_step_denoised(x_2, model, sigma_mid, sb, **extra_args)
            denoised_2 = denoised_2a * (sa ** 2) * 0.625 + denoised_2b * (sb ** 2) * 0.375  / (0.95**2)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_smea_multi_bs2_s(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    sample = sample_euler_smea_multi_bs2(model, x, sigmas, extra_args, callback, disable, s_churn, s_tmin, s_tmax, s_noise, smooth=True)
    return sample

@torch.no_grad()
def sample_euler_smea_multi_bs2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., smooth=False):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.167:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            scale = (sigmas[i] / sigmas[0]) ** 2
            scale = scale.item()
            sa = 1 - scale * 0.25
            sb = 1 + scale * 0.15
            sigA = sigma_mid / (sa ** 2)
            sigB = sigma_mid / (sb ** 2)
            denoised_2a = smea_sampling_step_denoised(x_2, model, sigA, sa, smooth, **extra_args)
            denoised_2b = smea_sampling_step_denoised(x_2, model, sigB, sb, smooth, **extra_args)
            denoised_2 = (denoised_2a * (sa ** 2) * 0.5 * sb ** 2 + denoised_2b * (sb ** 2) * 0.5 * sa ** 2)
            d_2 = to_d(x_2, sigA * 0.5 * sb ** 2 + sigB * 0.5 * sa ** 2, denoised_2)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_smea_multi_cs(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.167:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            scale = ((len(sigmas) - i) / len(sigmas)) ** 2
            sa = 1 - scale * 0.25
            denoised_2 = smea_sampling_step_denoised(x_2, model, sigma_mid, sa, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2 * (sa ** 2) * 1.25)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_smea_multi_as(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = k_diffusion.sampling.torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised}) 
        if sigmas[i + 1] > 0 and i < len(sigmas) * 0.167:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            scale = ((len(sigmas) - i) / len(sigmas)) ** 2
            sa = 1 + scale * 0.15
            denoised_2 = smea_sampling_step_denoised(x_2, model, sigma_mid, sa, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2 * (sa ** 2) * 0.75)
            x = x + d_2 * dt_2
        else:
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
    return x

## og sampler
@torch.no_grad()
def sample_euler_dy_og(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        # print(i)
        # i第一步为0
        gamma = max(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        # print(sigma_hat)
        dt = sigmas[i + 1] - sigma_hat
        if gamma > 0:
            x = x - eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = sampling.to_d(x, sigma_hat, denoised)
        if sigmas[i + 1] > 0:
            if i // 2 == 1:
                x = dy_sampling_step(x, model, dt, sigma_hat, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        # Euler method
        x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_smea_dy_og(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = max(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        dt = sigmas[i + 1] - sigma_hat
        if gamma > 0:
            x = x - eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = sampling.to_d(x, sigma_hat, denoised)
        # Euler method
        x = x + d * dt
        if sigmas[i + 1] > 0:
            if i + 1 // 2 == 1:
                x = dy_sampling_step(x, model, dt, sigma_hat, **extra_args)
            if i + 1 // 2 == 0:
                x = smea_sampling_step(x, model, dt, sigma_hat, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
    return x

## TCD

def sample_tcd_euler_a(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, gamma=0.3):
    # TCD sampling using modified Euler Ancestral sampler. by @laksjdjf
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        #d = to_d(x, sigmas[i], denoised)		
        sigma_from = sigmas[i]
        sigma_to = sigmas[i + 1]

        t = model.inner_model.sigma_to_t(sigma_from)
        down_t = (1 - gamma) * t
        sigma_down = model.inner_model.t_to_sigma(down_t)

        if sigma_down > sigma_to:
            sigma_down = sigma_to
        sigma_up = (sigma_to ** 2 - sigma_down ** 2) ** 0.5     

        # same as euler ancestral
        d = to_d(x, sigma_from, denoised)
        dt = sigma_down - sigma_from
        x += d * dt

        if sigma_to > 0 and gamma > 0:
            x =  x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigma_up
    return x

@torch.no_grad()
def sample_tcd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, gamma=0.3):
    # TCD sampling using modified DDPM.
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        sigma_from, sigma_to = sigmas[i], sigmas[i+1]

        # TCD offset, based on gamma, and conversion between sigma and timestep
        t = model.inner_model.sigma_to_t(sigma_from)
        t_s = (1 - gamma) * t
        sigma_to_s = model.inner_model.t_to_sigma(t_s)

        # if sigma_to_s > sigma_to:
        #     sigma_to_s = sigma_to
        # if sigma_to_s < 0:
        #     sigma_to_s = torch.tensor(1.0)
        #print(f"sigma_from: {sigma_from}, sigma_to: {sigma_to}, sigma_to_s: {sigma_to_s}")


        # The following is equivalent to the comfy DDPM implementation
        # x = DDPMSampler_step(x / torch.sqrt(1.0 + sigma_from ** 2.0), sigma_from, sigma_to, (x - denoised) / sigma_from, noise_sampler)

        noise_est = (x - denoised) / sigma_from
        x /= torch.sqrt(1.0 + sigma_from ** 2.0)

        alpha_cumprod = 1 / ((sigma_from * sigma_from) + 1)  # _t
        alpha_cumprod_prev = 1 / ((sigma_to * sigma_to) + 1) # _t_prev
        alpha = (alpha_cumprod / alpha_cumprod_prev)

        ## These values should approach 1.0?
        # print(f"alpha_cumprod: {alpha_cumprod}")
        # print(f"alpha_cumprod_prev: {alpha_cumprod_prev}")
        # print(f"alpha: {alpha}")


        # alpha_cumprod_down = 1 / ((sigma_to_s * sigma_to_s) + 1)  # _s
        # alpha_d = (alpha_cumprod_prev / alpha_cumprod_down)
        # alpha2 = (alpha_cumprod / alpha_cumprod_down)
        # print(f"** alpha_cumprod_down: {alpha_cumprod_down}")
        # print(f"** alpha_d: {alpha_d}, alpha2: #{alpha2}")

        # epsilon noise prediction from comfy DDPM implementation
        x = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise_est / (1 - alpha_cumprod).sqrt())
        # x = (1.0 / alpha_d).sqrt() * (x - (1 - alpha) * noise_est / (1 - alpha_cumprod).sqrt())

        first_step = sigma_to == 0
        last_step = i == len(sigmas) - 2

        if not first_step:
            if gamma > 0 and not last_step:
                noise = noise_sampler(sigma_from, sigma_to)

                # x += ((1 - alpha_d) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise
                variance = ((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * (1 - alpha_cumprod / alpha_cumprod_prev)
                x += variance.sqrt() * noise # scale noise by std deviation

                # relevant diffusers code from scheduling_tcd.py
                # prev_sample = (alpha_prod_t_prev / alpha_prod_s).sqrt() * pred_noised_sample + (
                #     1 - alpha_prod_t_prev / alpha_prod_s
                # ).sqrt() * noise

            x *= torch.sqrt(1.0 + sigma_to ** 2.0)

        # beta_cumprod_t = 1 - alpha_cumprod
        # beta_cumprod_s = 1 - alpha_cumprod_down

 
    return x
