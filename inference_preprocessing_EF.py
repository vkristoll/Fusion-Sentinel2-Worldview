#Pre-processing steps for the Early Fusion (EF) networks (Fusion-ResNet, Fusion-GAN)
#Creation of input inference file

#Imporing libraries
import numpy as np
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler
from osgeo import gdal

from training_preprocessing_EF import input_array, bandsin, scaler

#Read input file
''' The input of the Early Fusion (EF) networks (Fusion-ResNet, Fusion-GAN) was created by concatenating the S2 and WV-3 bands. During the inference stage, 
the fused output image (spatial resolution: 4 m) was created by feeding the network with i) the WV-3 bands with 4 m spatial resolution and ii) the S2 bands with 20 m
spatial resolution after upsampling to match the x-y size of 4 m resolution. The sequence of the bands matched the sequence of the corresponding wavelengths.   Thus, the spatial
resolution ratio between the WV-3 and S2 bands was 1/5. '''

im_inputinf=gdal.Open("S2WV_cat_input_inference.tiff")
input_arrayinf=np.array(im_inputinf.ReadAsArray()) 

shapein=np.shape(input_arrayinf)
rows=shapein[1]
cols=shapein[2]

'''Create new array where 1% of the histogram values for each band (left and right) are clipped to prevent lower performance 
due to sparse extreme values.'''
input_array_clip=np.uint16(np.zeros((rows,cols,bandsin)))

for i in range (bandsin):   
    # The histogram values of the training input image are used (input_array)
    p1 = np.percentile(input_array[i,:,:], 1)
    p99 = np.percentile(input_array[i,:,:], 99)    
    min_val=np.min(input_array[i,:,:])
    max_val=np.max(input_array[i,:,:])
    input_array_clip[:,:,i]= exposure.rescale_intensity(input_arrayinf[i,:,:], in_range=(p1, p99), out_range=(min_val,max_val))

#Reshape array of clipped values    
input_array_clip2=np.float16(np.zeros((rows*cols,bandsin)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        input_array_clip2[c,:]=input_array_clip[i,j,:]
        
##Create array with value range [0,1]. The scaling is performed according to the values of the training image. 
input_array_scale=scaler.fit_transform(input_array_clip2)

#Reshape array of normalized values to original shape
input_arrayfin=np.float16(np.zeros((rows,cols,bandsin)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        input_arrayfin[i,j,:]=input_array_scale[c,:]

#Save input inference file
target_layer="S2WV_cat_input_inference_prep.tiff "  
driver= gdal.GetDriverByName('GTiff')  
target_ds = driver.Create(target_layer, cols, rows, bands=bandsin, eType=gdal.GDT_Float32)        
    
for i in range(bandsin):            
    outBand = target_ds.GetRasterBand(i+1)
    outBand.WriteArray(input_arrayfin[:,:,i])
target_ds= None     










  
