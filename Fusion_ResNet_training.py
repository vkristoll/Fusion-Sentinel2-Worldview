#Training the Fusion-ResNet network

#Importing libraries
import numpy as np
from osgeo import gdal
import random
import time

import torch
from torch import nn
import torch.optim as optim

from training_patches_creation import create_patches
from Fusion_ResNet_model_definition import  ResBlock, Fusion_ResNet

#define path to save the weights
path_model="/define path to save weights"

#Define the device
device = torch.device("cuda:0")

#Read input training preprocessed file
im_input=gdal.Open("S2WV_cat_input_training_prep.tiff")
#Change the axes order from bands,rows,cols to rows,cols,bands
input_array=np.array(im_input.ReadAsArray()).transpose(1,2,0)  

shapein=np.shape(input_array)
rows=shapein[0]
cols=shapein[1]
bandsin=shapein[2]

#Read output file: The 20 m S2 image was fed to the network as output.
im_output=gdal.Open("S2_output_training_prep.tiff")
#Change the axes order from bands,rows,cols to rows,cols,bands
output_array=np.array(im_output.ReadAsArray()).transpose(1,2,0) 

shapeout=np.shape(output_array)
bandsout=shapeout[2]

#Create training patches
patch_size=9
[x,y]=create_patches(output_array,input_array,patch_size, bandsin, bandsout)

#Change the axes order from rows,cols,bands to bands,rows,cols 
x=x.transpose(0,3,1,2)
y=y.transpose(0,3,1,2)

#Calculate the number of patches
dataset_size=np.shape(x)[0]

#Define batch and patch sizes
batch_size=128
patch_size=40

#Define function that creates the training batches
def generate_batch():
    
    randomindex_list=random.sample(range(0,dataset_size),batch_size)
    
    X=np.float32(np.zeros((batch_size,bandsin,patch_size,patch_size))) 
    Y=np.float32(np.zeros((batch_size,bandsout,patch_size,patch_size)))
    
    c=0
    for i in range(0,batch_size,1):
       
        c=c+1
        X[c-1,:,:,:]=x[randomindex_list[i],:,:,:] 
        Y[c-1,:,:,:]=y[randomindex_list[i],:,:,:]
        
    return X,Y

#Define the loss function 
MSE= nn.MSELoss() 
#Define the model
model = Fusion_ResNet(bandsin, ResBlock).to(device)
#Define the optimizer
model_optim = optim.Adam(model.parameters(), lr = 0.0001)
   
# from torchsummary import summary
# summary(model,input_size= (bandsin,patch_size, patch_size))

#Create list to save the loss scores
model_losses = []
#Define the number of epochs and training steps
epochs = 800
train_steps=int(dataset_size/batch_size)

start_time = time.time()
for epoch in range(epochs):     
    loss_total=0      
    for i in range (train_steps): 
               
        model.zero_grad()
        
        #generate an input-output batch
        [datain,dataout]=generate_batch()
        # Convert the batch array to tensor
        datain_t=torch.tensor(datain).to(device) 
        dataout_t=torch.tensor(dataout).to(device)
        
        modelout=model(datain_t)
        model_loss=MSE(modelout,dataout_t)
        
        model_loss.backward()      
        model_optim.step() 
       
        loss_total +=  model_loss.item()
    
    loss_av=loss_total/train_steps    
    print('Epoch [{}/{}], loss_av: {:.6f}'.format(epoch, epochs, loss_av))
            
    model_losses.append(loss_av)     
    torch.save(model.state_dict(), f"{path_model}/resnet_{epoch+1}.pth")
    
print("--- %s seconds ---" % (time.time() - start_time))
timesave=time.time() - start_time    
model_losses.append(timesave)
np.savetxt("resnetlossaver.csv", model_losses,  header="  #resnet_losses ",fmt='%10.6f')

    
