#Inference, create 4m fused S2 image for the Siamese model (Fusion-PNN-Siamese)

#Importing libraries
import keras
from keras.models import Input, Model
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras import metrics

import numpy as np    
from osgeo import gdal

from siamese_model_definition import siamese_model

#Read input inference S2 file preprocessed
im_inputA=gdal.Open("S2_input_inference_prep.tiff")
#Change the axes order from bands,rows,cols to rows,cols,bands
inputA_array=np.array(im_inputA.ReadAsArray()).transpose(1,2,0)  

shapeinA=np.shape(inputA_array)
rows=shapeinA[0]
cols=shapeinA[1]
bandsinA=shapeinA[2]

#Read input inference WV file preprocessed 
im_inputB=gdal.Open("WV_input_inference_prep.tiff")
#Change the axes order from bands,rows,cols to rows,cols,bands
inputB_array=np.array(im_inputB.ReadAsArray()).transpose(1,2,0)  

shapeinB=np.shape(inputB_array)
bandsinB=shapeinB[2]

#Define the model
patch_size=9
bandsout=13

#Define the model
model=siamese_model(patch_size,bandsinA, bandsinB, bandsout)

#Load the weights
model.load_weights("weightsSiamesebest.h5") 

#Make predictions
a=int(cols/patch_size)
b=int(rows/patch_size)

#Create empty list to store predictions
l=[]

X= np.zeros((a,patch_size,patch_size,bandsinA))
Xb= np.zeros((a,patch_size,patch_size,bandsinB))
c=0
for i in range(0,rows,patch_size):    
    print(" The repetition number is %s" %i)     
    for j in range(0,cols,patch_size):
        c=c+1
        # X stores predictions equal to int(cols/input_patch_xysize)
        X[c-1,:,:,:]= inputA_array[i:i+patch_size,j:j+patch_size,:]
        Xb[c-1,:,:,:]= inputB_array[i:i+patch_size,j:j+patch_size,:]
    c=0
    predictions=model.predict([X,Xb])
    l.append(predictions)
    

#Create the output fused S2 image    
fused=np.float16(np.zeros((rows,cols,bandsout)))
for i in range(b):
    for j in range (a):
        fused[i*patch_size:i*patch_size+patch_size,j*patch_size:j*patch_size+patch_size,:]=l[i][j]
        
#Save the fused image
target_layer="Fusedsiamese.tiff"     
driver= gdal.GetDriverByName('GTiff')    

target_ds = driver.Create(target_layer, cols, rows, bands=bandsout, eType=gdal.GDT_Float32)      
    
for i in range(bandsout):            
    outBand = target_ds.GetRasterBand(i+1)
    outBand.WriteArray(fused[:,:,i])
target_ds= None




