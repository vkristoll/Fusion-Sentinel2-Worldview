#Inference, create 4m fused S2 image for the Fusion-GAN model

#Importing libraries
import numpy as np
from osgeo import gdal

import torch
from torch import nn
import torch.nn.functional as F

from Fusion_GAN_model_definition import Gen

#Define device
device = torch.device("cuda:0")

#Define the path to the saved weights
PATH_TO_MODEL="..../Gen_800.pth"

#Define the number of input and output bands for the generator 
gen_in_ch=21
gen_out_ch=13

#Define the number of feature maps for the first convolutional layer of the generator 
dim_g=32

#Read input inference preprocessed file
im_input=gdal.Open("S2WV_cat_input_inference_prep.tiff")
input_array=np.array(im_input.ReadAsArray())

shapein=np.shape(input_array)
rows=shapein[1]
cols=shapein[2]
bandsin=shapein[0]

#Define the generator model
model= Gen(gen_in_ch,dim_g,gen_out_ch).to(device)
#Load the weights
model.load_state_dict(torch.load(PATH_TO_MODEL))

#Define  the patch size
patch_size=40

#make predictions
a=int(cols/patch_size)
b=int(rows/patch_size)

#Create empty list to store predictions
l=[]

X= np.float32(np.zeros((a,gen_out_ch,patch_size,patch_size)))
c=0

with torch.no_grad():    
    model.eval()    
    for i in range(0,rows,patch_size):
        
        print(" The repetition number is %s" %i) 
        
        for j in range(0,cols,patch_size):
            c=c+1
            X[c-1,:,:,:]= input_array[:,i:i+patch_size,j:j+patch_size]
            X_t=torch.tensor(X).to(device) 
        c=0
        predictions=Gen(X_t).detach().cpu().numpy()      
        l.append(predictions)
        
#Create the output fused S2 image   
fused=np.float32(np.zeros((gen_out_ch,rows,cols)))  
for i in range(b):
    for j in range (a):
        fused[:,i*patch_size:i*patch_size+patch_size,j*patch_size:j*patch_size+patch_size]=l[i][j]
               
#Save the fused image
target_layer="Fused_Fusion_GAN.tiff"     
driver= gdal.GetDriverByName('GTiff')    

target_ds = driver.Create(target_layer, cols, rows, bands=gen_out_ch, eType=gdal.GDT_Float32)        
         
for i in range(gen_out_ch):            
    outBand = target_ds.GetRasterBand(i+1)
    outBand.WriteArray(fused[i,:,:])
target_ds= None
        
        
        
        
        
        