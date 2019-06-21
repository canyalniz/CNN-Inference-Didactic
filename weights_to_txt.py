# -*- coding: utf-8 -*-

import numpy as np

from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.layers import Dropout
from keras.layers import Input


inputs = Input(shape=(3,63,63))
c = Convolution2D(32, (3, 3), activation='relu', strides=2, data_format='channels_first')(inputs)
c = Dropout(0.1)(c)
c = MaxPooling2D(pool_size=(3,3), strides=2, data_format='channels_first')(c)
c = Convolution2D(32, (3, 3), activation='relu', padding='same', strides=1, data_format='channels_first')(c)
c = Dropout(0.1)(c)
c = MaxPooling2D(pool_size=(3,3), strides=2, data_format='channels_first')(c)
c = Flatten()(c)
c = Dense(units=512, activation='relu')(c)
c = Dropout(0.2)(c)
outputs = Dense(units=1, activation='linear')(c)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.load_weights('weights_vanilla.h5') #########################  .h5 WEIGHTS FILE GOES HERE

weights = []
config = []
for layer in model.layers:
    weights.append(layer.get_weights())
    config.append(layer.get_config())

########### ADJUSTMENT FOR THE DENSE LAYERS ###########
weights[8][0] = np.expand_dims(np.expand_dims(weights[8][0], 0), 0)
weights[10][0] = np.expand_dims(np.expand_dims(weights[10][0], 0), 0)


for l in [1,4,8,10]:
    for h in range(0,weights[l][0].shape[0]):
        for w in range(0,weights[l][0].shape[1]):
            file_w = 'layer'+str(l)+'_h'+str(h)+'_w'+str(w)+'_weights.txt'
            
            np.savetxt(file_w, weights[l][0][h][w], fmt='%.8e')

    file_b = 'layer'+str(l)+'_biases.txt'
    np.savetxt(file_b, weights[l][1], fmt='%.8e')