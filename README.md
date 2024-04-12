# sd-webui-smea
smea sampler for a1111 webui (single file only)

originally created by [Koishi-Star](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler) and [licyk](https://github.com/licyk/advanced_euler_sampler_extension)

Euler Max: from licyk's repo    
Euler Dy: Euler Dy but with DPM2 tweak, toggle on/off every step (stopped at 1/3 total steps)    
Euler Smea: Euler Smea but with DPM2 tweak, toggle on/off every step (stopped at 1/2 total steps)    
Euler Smea dyn: Euler Smea but with DPM2 tweak, scale down > normal > up > normal >... every step (stopped at 1/2 total steps)    
Euler Smea Dy: Euler Smea Dy but with DPM2 tweak, switch between folded to 1/2 size and scale up every step (stopped at 1/3 total steps)    
Euler Dy koishi-star: og Euler Dy made by koishi-star    
Euler Smea Dy koishi-star: og Euler Smea Dy made by koishi-star    
    
The reason of many experiments is due to og sampler tends to blurred the background or overfry image, so I checked DPM2 sampler and experiment if it's worth to tweak it
