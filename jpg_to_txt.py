# -*- coding: utf-8 -*-

import numpy as np
from keras.preprocessing import image

img = image.load_img('11500.jpg', target_size = (63, 63)) ###################### .jpg IMAGE FILE GOES HERE
img = image.img_to_array(img, data_format='channels_first')
np.savetxt('img_0.txt', img[0], fmt='%.8e')
np.savetxt('img_1.txt', img[1], fmt='%.8e')
np.savetxt('img_2.txt', img[2], fmt='%.8e')
