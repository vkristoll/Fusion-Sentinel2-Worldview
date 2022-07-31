#Training the Siamese model (Fusion-PNN-Siamese)

#Importing libraries
import keras
from keras.models import Input, Model
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras import metrics

import numpy as np
from osgeo import gdal
import time

from training_patches_creation import create_patches
from siamese_model_definition import siamese_model

#Read input training S2 file preprocessed
im_inputA=gdal.Open("S2_input_training_prep.tiff")
#Change the axes order from bands,rows,cols to rows,cols,bands
inputA_array=np.array(im_inputA.ReadAsArray()).transpose(1,2,0)  

shapeinA=np.shape(inputA_array)
rows=shapeinA[0]
cols=shapeinA[1]
bandsinA=shapeinA[2]

#Read input training WV file preprocessed 
im_inputB=gdal.Open("WV_input_training_prep.tiff")
#Change the axes order from bands,rows,cols to rows,cols,bands
inputB_array=np.array(im_inputB.ReadAsArray()).transpose(1,2,0)  

shapeinB=np.shape(inputB_array)
bandsinB=shapeinB[2]

#Read output file: The 20 m S2 image was fed to the network as output.
im_output=gdal.Open("S2_output_training_prep.tiff")
#Change the axes order from bands,rows,cols to rows,cols,bands
output_array=np.array(im_output.ReadAsArray()).transpose(1,2,0) 

shapeout=np.shape(output_array)
bandsout=shapeout[2]

#Create training patches
patch_size=9

[x,y]=create_patches(output_array,inputA_array,patch_size, bandsinA, bandsout)
[xb,y]=create_patches(output_array,inputB_array,patch_size, bandsinB, bandsout)

#Define the model
model=siamese_model(patch_size,bandsinA, bandsinB, bandsout)

#Save best weights , loss scores and training time
checkpoint=keras.callbacks.ModelCheckpoint("weightsSiamesebest.h5",monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
callbacks_list = [checkpoint]

start_time = time.time()
history=model.fit([x,xb],y,epochs=4000, batch_size =128,callbacks=callbacks_list)

print("--- %s seconds ---" % (time.time() - start_time))
timesave=time.time() - start_time

loss_values=history.history['mse']

loss_values.append(timesave)
np.savetxt("siameseloss.csv", loss_values,  header=" #loss  ",fmt='%10.6f')