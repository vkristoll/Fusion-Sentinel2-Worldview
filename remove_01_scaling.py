#Removing the [0,1] scaling. 

#Importing libraries
from osgeo import gdal
import numpy as np

from training_preprocessing_EF import scaler2

#Read fused image with [0,1] values
im_fused01=gdal.Open("im_fused01.tiff") 
im_fused01_array=np.array(im_fused01.ReadAsArray())

shape=np.shape(im_fused01_array)
rows=shape[1]
cols=shape[2]
bands=shape[0]

#Remove the 0-1 scaling transformation to recover uint16 values       
fused_uint16=np.float16(np.zeros((rows*cols,bands)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        fused_uint16[c,:]=im_fused01_array[i,j,:]    
        
fused_uint16= scaler2.inverse_transform(fused_uint16)  

#Reshape to the fused image shape
fused_uint16b=np.float16(np.zeros((rows,cols,bands)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        fused_uint16b[i,j,:]=fused_uint16[c,:]    
       
#Save the uint16 fused image  
target_layer="im_fused_uint16.tiff"     
driver= gdal.GetDriverByName('GTiff')    
target_ds = driver.Create(target_layer, cols, rows, bands, eType=gdal.GDT_UInt16)        
    
for i in range(bands):            
    outBand = target_ds.GetRasterBand(i+1)
    outBand.WriteArray(fused_uint16b[:,:,i])
target_ds= None


