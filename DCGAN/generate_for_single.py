
import os
import argparse 
import torch 
import torchvision.utils as vutils
import numpy as np
import matplotlib.animation as animation 
import random

from dcgan import Generator  

output_dir = 'generated_images'

os.makedirs(output_dir, exist_ok=True)
parser = argparse.ArgumentParser() 
parser.add_argument('-load_path', default='model/model_final.pth', help='Checkpoint to load path from') 
parser.add_argument('-num_output', default=64, help='Number of generated outputs') 
args = parser.parse_args() 


state_dict = torch.load(args.load_path) 


device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")


params = state_dict['params'] 


netG = Generator(params).to(device) 


netG.load_state_dict(state_dict['generator']) 

print(netG)

print(args.num_output) 

noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)


with torch.no_grad():
	
    generated_img = netG(noise).detach().cpu() 
# multipple images
for i in range(args.num_output):
    image_path = os.path.join(output_dir, f'generated_image_{i}.png')
    plt.imshow(np.transpose(generated_img[i], (1, 2, 0)))
    plt.axis("off")
    plt.savefig(image_path, dpi=600)
    plt.close()

print(f"Generated images saved in {output_dir} directory.")