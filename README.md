# sd-webui-smea
smea sampler for a1111 webui (in single file only)

originally created by [Koishi-Star](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler) and [ananosleep](https://github.com/ananosleep/advanced_euler_sampler_extension)

![sample](https://github.com/AG-w/sd-webui-smea/blob/main/sample.jpg?raw=true)
   
*Euler Dy: Euler Dy with DPM2 tweak, toggle on/off every step    
*Euler Smea: Euler Smea with DPM2 tweak, toggle on/off every step    
*Euler Smea Dy: Euler Smea Dy with DPM2 tweak, loopping scale up > folded to 1/2 size > normal >... every step     
*Euler Smea dyn a: Euler Smea with DPM2 tweak (less sigma), toggle on/off (scale up) every step every step    
*Euler Smea dyn b: Euler Smea with DPM2 tweak (less sigma), loopping scale down > up > normal >... every step   
*Euler Smea dyn c: Euler Smea with DPM2 tweak (less sigma), toggle on/off (scale down) every step every step   
all sampler above stopped smea / dy sampling at 1/3 total steps    
    
*Euler Max: from ananosleep's repo     
*Euler Dy koishi-star: og Euler Dy made by koishi-star        
*Euler Smea Dy koishi-star: og Euler Smea Dy made by koishi-star     
       
The reason of many experiments is due to og sampler tends to blurred the background or overfry the image, so I checked DPM2 sampler and experiment if it's worth to tweak it
    
Explanation:    
What Smea sampling do is scaling latent image > denoise > scale it back to original size    
What dy sampling do is shrinking latent image to 1/2 size > denoise > extend it to original size    
since what they did is bascially just scaling latent image, I use smea sampling only    
What all these samplers do is bascailly trying to combine different scaled latent image to generate image with better detail
