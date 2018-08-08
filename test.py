from keras.models import Model
import numpy as np
from keras.layers import Input
from model import darknet_body

image_input = Input(shape = (416, 416, 3))
output = darknet_body(image_input)
model = Model(image_input, output)
model.load_weights('./darknet53_weights.h5', by_name = True, skip_mismatch = True)
weight = model.get_weights()
np.save('test.npy', weight)