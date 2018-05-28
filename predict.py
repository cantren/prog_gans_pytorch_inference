#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Sample code for inference of Progressive Growing of GANs paper
(https://github.com/tkarras/progressive_growing_of_gans)
using a CelebA snapshot
"""
"""
PyTorch implementation by github.com/ptrblck
"""
"""
"Random walk across the latent space" mods made to ptrblck's code by github.com/cantren
"""

from __future__ import print_function
import argparse

import torch
from torch.autograd import Variable

from model import Generator

from utils import scale_image

import matplotlib.pyplot as plt
import cv2 # added some imports
import random # added some imports

parser = argparse.ArgumentParser(description='Inference demo')
parser.add_argument(
    '--weights',
    default='100_celeb_hq_network-snapshot-010403.pth',
    type=str,
    metavar='PATH',
    help='path to PyTorch state dict')
parser.add_argument('--cuda', dest='cuda', action='store_true')

#seed = 2808 # got tired of modifying the seed by hand
seed = random.SystemRandom().randint(1000, 3000) # got tired of modifying the seed by hand
use_cuda = False

torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

def run(args, x):

    global use_cuda
    
    #print('Loading Generator') # modified for readability
    model = Generator()
    model.load_state_dict(torch.load(args.weights))
    
    # Generate latent vector
#    x = torch.randn(1, 512, 1, 1) # now generated in main and passed as an argument
    

    if use_cuda:
        model = model.cuda()
        x = x.cuda()
    x = Variable(x, volatile=True)
    
    #print('Executing forward pass')
    images = model(x)
    
    if use_cuda:
        images = images.cpu()
    
#   
    images_np = images.data.numpy().transpose(0, 2, 3, 1)
    image_np = scale_image(images_np[0, ...])
    
    #print('Output')
    #plt.figure()
    #plt.imshow(image_np)
    cv2image_np = image_np[...,::-1]
    #cv2.imshow("test", cv2image_np)
    cv2.imwrite("/home/ubuntu/prog_gans_pytorch_inference/test.png",cv2image_np)
    #print(images_np.shape)



def main():
    global use_cuda
    args = parser.parse_args()

    if not args.weights:
        print('No PyTorch state dict path privided. Exiting...')
        return
    
    if args.cuda:
        use_cuda = True
    rvector = torch.randn(1, 512, 1, 1)
    loop = 0;
    v0 = random.sample(range(0, 511), 4)
    v1 = v0[0]
    v2 = v0[1]
    v3 = v0[2]
    v4 = v0[3]
    #print(str(v1))
    #print(str(v2))
    #print(str(v3))
    #print(str(v4)) 
    '''
    v1 = random.randint(0,511)
    v2 = random.randint(0,511)    
    v3 = random.randint(0,511)
    v4 = random.randint(0,511)
    '''    
    print('latent vector: '+ ' (Shape is [1][512])')
        
    while True:
        if loop % 5 == 0:
            v1 = random.randint(0,511)
            v2 = random.randint(0,511)
            v3 = random.randint(0,511)
            v4 = random.randint(0,511)
        if loop % 50 == 0:
            rvector = torch.randn(1, 512, 1, 1)
        temp = rvector[0,:,0,0]
        u1 = random.uniform(0,2)-1.0
        temp[v1] = temp[v1] + u1

        u2 = random.uniform(0,2)-1.0
        temp[v2] = temp[v2] + u2

        u3 = random.uniform(0,2)-1.0
        temp[v3] = temp[v3] + u3

        u4 = random.uniform(0,2)-1.0
        temp[v4] = temp[v4] + u4

        vec2 = sorted(vec1)
        vec3 = [vec1[vec2[0]],vec1[vec2[1]],vec1[vec2[2]],vec1[vec2[3]]]
        vec4 = [temp[vec2[0]],temp[vec2[1]],temp[vec2[2]],temp[vec2[3]]]

        print('dim:   [... ,   ' + str(vec2[0]) +  ' , ... ,   ' + str(vec2[1]) + ' , ... ,   ' + str(vec2[2]) + ' , ... ,   ' + str(vec2[3]) + ' , ...]')
        print("delta: [... , %s , ...]"%" , ... , ".join("%.2f" % f for f in vec3))
        print("value: [... , %s , ...]"%" , ... , ".join("%.2f" % f for f in vec4))
        
        
        xr = rvector
        xr[0,:,0,0] = temp
        run(args, xr) # modified to allow for the randomized vector to be passed as an argument
        loop = loop + 1

if __name__ == '__main__':
    main() 
