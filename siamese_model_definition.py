#Define the Siamese model

#Importing libraries
import keras
from keras.models import Input, Model
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras import metrics

def siamese_model(patch_size, bandsinput1, bandsinput2, bandsoutput, l_r = 0.0001): 
   
    
    img_input1 = Input(shape = (patch_size, patch_size, bandsinput1))
    img_input2 = Input(shape = (patch_size, patch_size, bandsinput2))
    
    conv1_1 = Conv2D(64, (9, 9), padding = 'same', activation = 'relu')(img_input1)    
    conv1_2=  Conv2D(64, (9, 9), padding = 'same', activation = 'relu')(img_input2)
    
    conv2_1 = Conv2D(32, (5, 5), padding = 'same', activation = 'relu')(conv1_1)    
    conv2_2 = Conv2D(32, (5, 5), padding = 'same', activation = 'relu')(conv1_2)   
    
    merged = concatenate([conv2_1, conv2_2])
    
    conv3 = Conv2D(bandsoutput, (5, 5), padding = 'same',activation = 'sigmoid' )(merged)
    
       
    model = Model(inputs = [img_input1, img_input2], outputs = conv3)
    model.compile(optimizer = Adam(lr = l_r), loss = 'mse', metrics = ['mse'])   
    
    return model

