#Pre-processing steps for the Siamese network (Fusion-PNN-Siamese)
#Creation of input and output training files

#Imporing libraries
import numpy as np
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler
from osgeo import gdal

''' In Fusion-PNN-Siamese, the S2 and WV-3 bands were fed as input to two different branches. During training the inputs of the CNN were:
i) the WV-3 bands downsampled to 20 m spatial resolution and ii) the Sentinel-2 bands downsampled to 100 m and then upsampled to 20 m. Thus, the spatial
resolution ratio between the WV-3 and S2 bands was 1/5.'''

#Read first branch input (S2 file)
im_inputA=gdal.Open("S2_input_training.tiff")
inputA_array=np.array(im_inputA.ReadAsArray()) 

shapeinA=np.shape(inputA_array)
rows=shapeinA[1]
cols=shapeinA[2]
bandsinA=shapeinA[0]

'''Create new array where 1% of the histogram values for each band (left and right) are clipped to prevent lower performance 
due to sparse extreme values.'''
inputA_array_clip=np.uint16(np.zeros((rows,cols,bandsinA)))

for i in range (bandsinA):    
    p1 = np.percentile(inputA_array[i,:,:], 1)
    p99 = np.percentile(inputA_array[i,:,:], 99)    
    min_val=np.min(inputA_array[i,:,:])
    max_val=np.max(inputA_array[i,:,:])
    inputA_array_clip[:,:,i]= exposure.rescale_intensity(inputA_array[i,:,:], in_range=(p1, p99), out_range=(min_val,max_val))

#Reshape array of clipped values    
inputA_array_clip2=np.float16(np.zeros((rows*cols,bandsinA)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        inputA_array_clip2[c,:]=inputA_array_clip[i,j,:]
        
#Create array with value range [0,1]     
scalerA=MinMaxScaler(feature_range = (0, 1))
inputA_array_scale=scalerA.fit_transform(inputA_array_clip2)

#Reshape array of normalized values to original shape
inputA_arrayfin=np.float16(np.zeros((rows,cols,bandsinA)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        inputA_arrayfin[i,j,:]=inputA_array_scale[c,:]        
        
#Read second branch input (WV file) 
im_inputB=gdal.Open("WV_input_training.tiff")
inputB_array=np.array(im_inputB.ReadAsArray()) 

shapeinB=np.shape(inputB_array)
bandsinB=shapeinB[0]

'''Create new array where 1% of the histogram values for each band (left and right) are clipped to prevent lower performance 
due to sparse extreme values.'''
inputB_array_clip=np.uint16(np.zeros((rows,cols,bandsinB)))

for i in range (bandsinB):    
    p1 = np.percentile(inputB_array[i,:,:], 1)
    p99 = np.percentile(inputB_array[i,:,:], 99)    
    min_val=np.min(inputB_array[i,:,:])
    max_val=np.max(inputB_array[i,:,:])
    inputB_array_clip[:,:,i]= exposure.rescale_intensity(inputB_array[i,:,:], in_range=(p1, p99), out_range=(min_val,max_val))

#Reshape array of clipped values    
inputB_array_clip2=np.float16(np.zeros((rows*cols,bandsinB)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        inputB_array_clip2[c,:]=inputB_array_clip[i,j,:]
        
#Create array with value range [0,1]     
scalerB=MinMaxScaler(feature_range = (0, 1))
inputB_array_scale=scalerB.fit_transform(inputB_array_clip2)

#Reshape array of normalized values to original shape
inputB_arrayfin=np.float16(np.zeros((rows,cols,bandsinB)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        inputB_arrayfin[i,j,:]=inputB_array_scale[c,:]
       
#Read output file: The 20 m S2 image was fed to the network as output.
im_output=gdal.Open("S2_output_training.tiff")
output_array=np.array(im_output.ReadAsArray())

shapeout=np.shape(output_array)
bandsout=shapeout[0]

'''Create new array where 1% of the histogram values for each band (left and right) are clipped to prevent lower performance 
due to sparse extreme values.'''

output_array_clip=np.uint16(np.zeros((rows,cols,bandsout)))

for i in range (bandsout):    
    p1 = np.percentile(output_array[i,:,:], 1)
    p99 = np.percentile(output_array[i,:,:], 99)    
    min_val=np.min(output_array[i,:,:])
    max_val=np.max(output_array[i,:,:])
    output_array_clip[:,:,i]= exposure.rescale_intensity(output_array[i,:,:], in_range=(p1, p99), out_range=(min_val,max_val))
    
#Reshape array of clipped values  
output_array_clip2=np.float16(np.zeros((rows*cols,bandsout)))

c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        output_array_clip2[c,:]=output_array_clip[i,j,:]

#Create array with value range [0,1]     
scaler2=MinMaxScaler(feature_range = (0, 1))
output_array_scale=scaler2.fit_transform(output_array_clip2)
    
#Reshape array of normalized values to original shape
output_arrayfin=np.float16(np.zeros((rows,cols,bandsout)))
c=-1
for i in range(rows):
    for j in range(cols):
        c=c+1
        output_arrayfin[i,j,:]=output_array_scale[c,:]
        
#Save input training S2 file
target_layer="S2_input_training_prep.tiff "  
driver= gdal.GetDriverByName('GTiff')  
target_ds = driver.Create(target_layer, cols, rows, bands=bandsinA, eType=gdal.GDT_Float32)        
    
for i in range(bandsinA):            
    outBand = target_ds.GetRasterBand(i+1)
    outBand.WriteArray(inputA_arrayfin[:,:,i])
target_ds= None     

#Save input training WV file
target_layer="WV_input_training_prep.tiff "  
driver= gdal.GetDriverByName('GTiff')  
target_ds = driver.Create(target_layer, cols, rows, bands=bandsinB, eType=gdal.GDT_Float32)        
    
for i in range(bandsinB):            
    outBand = target_ds.GetRasterBand(i+1)
    outBand.WriteArray(inputB_arrayfin[:,:,i])
target_ds= None   

#Save output training file
target_layer="S2_output_training_prep.tiff"  
driver= gdal.GetDriverByName('GTiff')  
target_ds = driver.Create(target_layer, cols, rows, bands=bandsout, eType=gdal.GDT_Float32)        
    
for i in range(bandsout):            
    outBand = target_ds.GetRasterBand(i+1)
    outBand.WriteArray(output_arrayfin[:,:,i])
target_ds= None  

