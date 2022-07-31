#Create input and output arrays of training patches

#input_arrayfin, output_arrayfin,  bandsin, bandsout are explained in training_preprocessing.py

import math
import numpy as np

def create_patches(output_arrayfin, input_arrayfin, patch_size, bandsin, bandsout):
     
     #calculate size of padded image (it should be divided with the patch_size without remainder)   
     
     rows=output_arrayfin.shape[0]
     cols=output_arrayfin.shape[1]    
     
     y_size=(math.floor(rows/patch_size) + 1) * patch_size
     x_size=(math.floor(cols/patch_size) + 1) * patch_size
     
     y_pad= int(y_size - rows)     
     x_pad= int(x_size - cols)
    
     #create padded  input and output images
     input_arraypad=np.float16(np.zeros((rows+y_pad, cols+x_pad, bandsin)))    
     input_arraypad[0:rows,0:cols,:]=input_arrayfin
     
     output_arraypad=np.float16(np.zeros((rows+y_pad, cols+x_pad, bandsout)))    
     output_arraypad[0:rows,0:cols,:]=output_arrayfin
     
     #create input and output training patches
     input_list_patches=[]
     output_list_patches=[]     
     
     for i in range(0, y_size-patch_size, 4): #~50% overlap
        for j in range(0, x_size-patch_size, 4):
            input_list_patches.append(input_arraypad[i:i+patch_size, j:j+patch_size,:])
            output_list_patches.append(output_arraypad[i:i+patch_size, j:j+patch_size,:])
     
     input_patches=np.zeros((len(input_list_patches),patch_size,patch_size,bandsin))
     output_patches=np.zeros((len(output_list_patches),patch_size,patch_size,bandsout))
     
     for i in range (len(input_list_patches)):
        input_patches[i,:,:,:] = input_list_patches[i]
        output_patches[i,:,:,:] = output_list_patches[i]   
    
     return input_patches, output_patches
   




