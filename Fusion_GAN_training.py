#Training the Fusion-GAN network

#Importing libraries
import numpy as np
from osgeo import gdal
import random
import time
import matplotlib.pyplot as plt

import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils

from training_patches_creation import create_patches
from Fusion_GAN_model_definition import Gen, Disc

#Define the paths to store the model weights and samples of training generator outputs
path_model="/path_model"
pred_train_path="/pred_train"

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

#Define the number of input and output bands for the generator and the discriminator
gen_in_ch=bandsin
gen_out_ch=bandsout
dim_in_ch=bandsout

#Define the number of feature maps for the first convolutional layer of the generator and the discriminator
dim_g=32
dim_d=32

#Define loss functions
BCE = nn.BCELoss()
L1 = nn.L1Loss() 

#Define the generator and discriminator models
Gen = Gen(gen_in_ch,dim_g,gen_out_ch).to(device)
Disc = Disc(dim_in_ch,dim_d).to(device)

#Define the optimizers
Gen_optim = optim.Adam(Gen.parameters(), lr=2e-4)
Disc_optim = optim.Adam(Disc.parameters(), lr=2e-4)

#Define lists to store loss scores
Disc_losses =[] 
Gen_losses = []
L1_losses =[]
r_gan_losses = []
f1_gan_losses=[]
f2_gan_losses=[]

#Define the number of epochs and training steps
epochs = 800
train_steps=int(dataset_size/batch_size)

L1_lambda = 100
time_list=[]
start_time = time.time()
for epoch in range(epochs):    
    Disc_loss_total=0
    Gen_loss_total=0
    L1_loss_total=0
    r_gan_loss_total=0
    f1_gan_loss_total=0
    f2_gan_loss_total=0       
    for i in range (train_steps):     
               
        r_masks = torch.ones(batch_size,1,2,2).to(device)
        f_masks = torch.zeros(batch_size,1,2,2).to(device)
        
        # generate a batch of images for input to the generator and discriminator
        [datain,realout]=generate_batch()
        datain_t=torch.tensor(datain).to(device) 
        realout_t=torch.tensor(realout).to(device)
        
        # disc
        Disc.zero_grad()        
        #real_patch
        r_patch=Disc(realout_t)
        r_gan_loss=BCE(r_patch,r_masks)           
        #fake_patch
        fake=Gen(datain_t)
        f_patch=Disc(fake.detach())
        f1_gan_loss=BCE(f_patch,f_masks)                    
        
        Disc_loss = r_gan_loss + f1_gan_loss
        Disc_loss.backward()
        Disc_optim.step()
        
        # gen
        Gen.zero_grad()
        f_patch = Disc(fake)
        f2_gan_loss=BCE(f_patch,r_masks)
        L1_loss = L1(fake,realout_t)
        Gen_loss = f2_gan_loss + L1_lambda*L1_loss
        Gen_loss.backward()    
        Gen_optim.step()
        
        Disc_loss_total += Disc_loss.item()
        Gen_loss_total+=Gen_loss.item()
        L1_loss_total+=L1_loss.item()
        
        r_gan_loss_total+=r_gan_loss.item()
        f1_gan_loss_total+=f1_gan_loss.item()
        f2_gan_loss_total+=f2_gan_loss.item()
                 
    Disc_loss_av=Disc_loss_total/train_steps
    Gen_loss_av=Gen_loss_total/train_steps
    L1_loss_av=L1_loss_total/train_steps
    
    r_gan_loss_av=r_gan_loss_total/train_steps
    f1_gan_loss_av=f1_gan_loss_total/train_steps
    f2_gan_loss_av=f2_gan_loss_total/train_steps
       
    print('Epoch [{}/{}],  Disc_loss: {:.6f}, \
                  Gen_loss: {:.6f},L1_loss: {:.6f}, r_gan_loss:{:.6f}, f1_gan_loss:{:.6f}, \
                      f2_gan_loss:{:.6f}'.format(epoch, epochs,  Disc_loss_av, Gen_loss_av, L1_loss_av, \
                          r_gan_loss_av, f1_gan_loss_av, f2_gan_loss_av))
            
    Disc_losses.append(Disc_loss_av)              
    Gen_losses.append(Gen_loss_av)                                                   
    L1_losses.append(L1_loss_av)
 
    r_gan_losses.append(r_gan_loss_av)
    f1_gan_losses.append(f1_gan_loss_av)
    f2_gan_losses.append(f2_gan_loss_av)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    timesave=time.time() - start_time 
    
    time_list.append(timesave)
    
    #Store sample outputs of the generator during training
    with torch.no_grad():
            Gen.eval()
            fakeim=Gen(datain_t)[0].detach().cpu()               
            Gen.train()
    figs=plt.figure(figsize=(10,10))
        
    A=realout_t[0].detach().cpu()           
    B=torch.zeros(3,patch_size,patch_size)
    C=torch.zeros(3,patch_size,patch_size)
            
    B[0,:,:]=A[3,:,:]
    B[1,:,:]=A[2,:,:]
    B[2,:,:]=A[1,:,:]  
            
    C[0,:,:]=fakeim[3,:,:]
    C[1,:,:]=fakeim[2,:,:]
    C[2,:,:]=fakeim[1,:,:]          
            
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("input image234")
    plt.imshow(np.transpose(vutils.make_grid(B, nrow=2, padding=5, 
            normalize=True).cpu(), (1,2,0)))
                   
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("predicted image234")
    plt.imshow(np.transpose(vutils.make_grid(C, nrow=2, padding=5, 
            normalize=True).cpu(), (1,2,0)))
      
    plt.savefig(os.path.join(pred_train_path ,"GANfusion"+"-"+str(epoch) +".png"))
    plt.close()
   
    #Store the weights   
    torch.save(Gen.state_dict(), f"{path_model}/Gen_{epoch}.pth")
    torch.save(Disc.state_dict(), f"{path_model}/Disc_{epoch}.pth")

    # Store loss scores and time
    list_all = [[i, j, k, l, m, n,o] for i, j , k, l, m, n,o in zip(Disc_losses,Gen_losses, L1_losses, r_gan_losses, f1_gan_losses,  f2_gan_losses, time_list)]
    list_all2= np.array(list_all)       
    
np.savetxt("metricsGAN.csv", list_all2, delimiter=',', header="  #Disc_losses, #Gen_losses , #L1_losses, #r_gan_losses,  #f1_gan_losses,  #f2_gan_losses , #time ",fmt='%10.6f')

            
   
         