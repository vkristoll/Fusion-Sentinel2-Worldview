#Pre-processing steps for the Siamese network (Fusion-PNN-Siamese)
#Creation of input inference files

#Imporing libraries
import numpy as np
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler
from osgeo import gdal

from training_preprocessing_Siamese import inputA_array,bandsinA,scalerA, bandsinB, inputB_array, scalerB

''' In Fusion-PNN-Siamese, the S2 and WV-3 bands were fed as input to two different branches. During the inference stage, the fused output
image (spatial resolution: 4 m) was created by feeding the network with i) the WV-3 bands with 4 m spatial resolution and ii) the S2 bands with 20 m
spatial resolution after upsampling to match the x-y size of 4 m resolution.  Thus, the spatial resolution ratio between the WV-3 and S2 bands was 1/5.'''

#Read first branch input (S2 file)
im_inputAinf=gdal.Open("S2_input_inference.tiff")
inputA_arrayinf=np.array(im_inputAinf.ReadAsArray()) 

shapeinA=np.shape(inputA_arrayinf)
rows=shapeinA[1]
cols=shapeinA[2]

'''Create new array where 1% of the histogram values for each band (left and right) are clipped to prevent lower performance 
due to sparse extreme values.'''
inputA_array_clip=np.uint16(np.zeros((rows,cols,bandsinA)))

for i in range (bandsinA):    
    # The histogram values of the training input image are used (inputA_array)
    p1 = np.percentile(inputA_array[i,:,:], 1)
    p99 = np.percentile(inputA_array[i,:,:], 99)    
    min_val=np.min(inputA_array[i,:,:])
    max_val=np.max(inputA_array[i,:,:])
    inputA_array_clip[:,:,i]= exposure.rescale_intensity(inputA_arrayinf[i,:,:], in_range=(p1, p99), out_range=(min_val,max_val))

#Reshape array of clipped values    
inputA_array_clip2=np.float16(np.zeros((rows*cols,bandsinA)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        inputA_array_clip2[c,:]=inputA_array_clip[i,j,:]
        
##Create array with value range [0,1]. The scaling is performed according to the values of the training image. 
inputA_array_scale=scalerA.transform(inputA_array_clip2)

#Reshape array of normalized values to original shape
inputA_arrayfin=np.float16(np.zeros((rows,cols,bandsinA)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        inputA_arrayfin[i,j,:]=inputA_array_scale[c,:]        
        
#Read second branch input (WV file) 
im_inputBinf=gdal.Open("WV_input_inference.tiff")
inputB_arrayinf=np.array(im_inputBinf.ReadAsArray()) 

'''Create new array where 1% of the histogram values for each band (left and right) are clipped to prevent lower performance 
due to sparse extreme values.'''
inputB_array_clip=np.uint16(np.zeros((rows,cols,bandsinB)))

for i in range (bandsinB):    
    # The histogram values of the training input image are used (inputA_array)
    p1 = np.percentile(inputB_array[i,:,:], 1)
    p99 = np.percentile(inputB_array[i,:,:], 99)    
    min_val=np.min(inputB_array[i,:,:])
    max_val=np.max(inputB_array[i,:,:])
    inputB_array_clip[:,:,i]= exposure.rescale_intensity(inputB_arrayinf[i,:,:], in_range=(p1, p99), out_range=(min_val,max_val))

#Reshape array of clipped values    
inputB_array_clip2=np.float16(np.zeros((rows*cols,bandsinB)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        inputB_array_clip2[c,:]=inputB_array_clip[i,j,:]
        
#Create array with value range [0,1]. The scaling is performed according to the values of the training image. 
inputB_array_scale=scalerB.transform(inputB_array_clip2)

#Reshape array of normalized values to original shape
inputB_arrayfin=np.float16(np.zeros((rows,cols,bandsinB)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        inputB_arrayfin[i,j,:]=inputB_array_scale[c,:]
    
#Save input inference S2 file
target_layer="S2_input_inference_prep.tiff "  
driver= gdal.GetDriverByName('GTiff')  
target_ds = driver.Create(target_layer, cols, rows, bands=bandsinA, eType=gdal.GDT_Float32)        
    
for i in range(bandsinA):            
    outBand = target_ds.GetRasterBand(i+1)
    outBand.WriteArray(inputA_arrayfin[:,:,i])
target_ds= None     

#Save input training WV file
target_layer="WV_input_inference_prep.tiff "  
driver= gdal.GetDriverByName('GTiff')  
target_ds = driver.Create(target_layer, cols, rows, bands=bandsinB, eType=gdal.GDT_Float32)        
    
for i in range(bandsinB):            
    outBand = target_ds.GetRasterBand(i+1)
    outBand.WriteArray(inputB_arrayfin[:,:,i])
target_ds= None   



