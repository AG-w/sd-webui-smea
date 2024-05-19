# sd-webui-smea
smea sampler experiments for a1111 webui    
These sampler has nothing to do with NAI's sampler or Euler sampler, I'm just suck at naming them.      
(smea here stands for "Shovel More Extra Artifacts")      
originally created by [Koishi-Star](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler) and [ananosleep](https://github.com/ananosleep/advanced_euler_sampler_extension)      
TCD sampler from [dfl](https://github.com/dfl/comfyui-tcd-scheduler)       
    
![sample2](https://github.com/AG-w/sd-webui-smea/blob/main/sample2.jpg?raw=true)    
![sample](https://github.com/AG-w/sd-webui-smea/blob/main/sample.jpg?raw=true)

**RECOMMEND: Use Smea mbs2 or Smea mds2, they add details (or artifacts) more reliably**    

Also check [Dynamic Thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding), you can add more details    

Euler Dy: og Euler Dy with DPM2 tweak, toggle on/off every step    
Euler Smea: og Euler Smea Dy with DPM2 tweak, use smea sampling only, toggle on/off every step    
Euler Smea Dy: og Euler Smea Dy with DPM2 tweak, loopping scale up > folded to 1/2 size > normal >... every step     
Euler Smea dyn a: Euler Smea with DPM2 tweak (less sigma), toggle on/off (scale up) every step every step    
Euler Smea dyn b: Euler Smea with DPM2 tweak (less sigma), loopping scale down > up > normal >... every step   
Euler Smea dyn c: Euler Smea with DPM2 tweak (less sigma), toggle on/off (scale down) every step every step   
Euler Smea md: Euler Smea with DPM2 tweak (less sigma), start with Smea mc then toggle Smea mb on/off every step, ended with Smea ma  
all sampler above stopped smea / dy sampling at 1/3 total steps      
     
Euler Smea ma: Euler Smea with DPM2 tweak (less sigma), combine scaled up latent image with normal one    
Euler Smea mb: Euler Smea with DPM2 tweak (less sigma), combine scaled up and scaled down latent image with normal one    
Euler Smea mc: Euler Smea with DPM2 tweak (less sigma), combine scaled down latent image with normal one          
Euler Smea mas: Euler Smea ma tweaked    
Euler Smea mbs: Euler Smea mb tweaked    
Euler Smea mcs: Euler Smea mc tweaked    
Euler Smea mds: Euler Smea md tweaked      
Euler Smea mbs2: Euler Smea mds with tweaked sigma          
Euler Smea mds2: Euler Smea mds with tweaked sigma     
Euler Smea mds2 max: Euler Smea mds2 with adjusted cosine wave      
Euler Smea mbs2 s: Euler Smea mbs2 with smoothed latent in process        
Euler Smea mds2 s: Euler Smea mds2 with smoothed latent in process     
all sampler above stopped smea sampling at 1/6 total steps    
    
Euler Max: from ananosleep's repo    
Euler Max2: Euler Max with adjusted cosine wave      
Euler Dy koishi-star: og Euler Dy made by koishi-star        
Euler Smea Dy koishi-star: og Euler Smea Dy made by koishi-star     
TCD and TCD Euler a: from dfl's repo        
       
### Explanation:    
The reason of many experiments is due to og sampler tends to blurred the background or overfry the image,    
so I checked DPM2 sampler and experiment if it's worth to tweak it    
What Smea sampling do is scaling latent image > denoise > scale it back to original size    
What dy sampling do is shrinking latent image to 1/2 size > denoise > extend it to original size    
since what they did is bascially just scaling latent image, I use smea sampling only    
What all these samplers do is bascailly trying to combine different scaled latent image to denoise image to generate better details (artifacts)
