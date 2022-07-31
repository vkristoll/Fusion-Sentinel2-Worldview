#Removing the 1% histogram clipping

#Importing libraries
from osgeo import gdal
import numpy as np
from skimage import exposure

from training_preprocessing_EF import output_array

#Read fused image with uint16 values
im_fuseduint16=gdal.Open("im_fused_uint16.tiff") 
im_fuseduint16_array=np.array(im_fuseduint16.ReadAsArray())

shape=np.shape(im_fuseduint16)
rows=shape[1]
cols=shape[2]
bands=shape[0]

#Remove the 1% histogram clipping
fused_noclip=np.zeros((bands,rows,cols))

for i in range (bands):    
    # The histogram values of the training output image are used (output_array)
    p1b = np.percentile(output_array[i,:,:], 1)
    p99b = np.percentile(output_array[i,:,:], 99)    
    min_valb=np.min(output_array[i,:,:])
    max_valb=np.max(output_array[i,:,:])    
    fused_noclip[i,:,:]= exposure.rescale_intensity(im_fuseduint16_array[i,:,:], in_range=(min_valb,max_valb), out_range=(p1b, p99b))

#Save the uint16 not clipped fused image 
target_layer="im_fused_uint16_noclip.tiff"     
driver= gdal.GetDriverByName('GTiff')   
target_ds = driver.Create(target_layer, cols, rows, bands, eType=gdal.GDT_Float32)        
    
for i in range(bands):            
    outBand = target_ds.GetRasterBand(i+1)
    outBand.WriteArray(fused_noclip[i,:,:])
target_ds= None

