import os
import numpy as np
import tensorflow as tf
prediction = tf.placeholder(shape = [1, 13, 13, 3, 85], dtype = tf.float32)
confidence = tf.sigmoid(prediction[..., 4:5])
class_prob = tf.sigmoid(prediction[..., 5:])
mask = [False]
mask_tensor = tf.placeholder(shape = [1], dtype = tf.bool)
result = [[2, 3, 4], [5, 6, 7]]
result = np.array(result)
print(result)
print(result.tolist())
print(os.path.join('../model_data', 'train' + '.tfrecords'))
# print(result[:,0])
# result_tensor = tf.placeholder(shape = [1, 3], dtype = tf.int32)
# result_mask = tf.boolean_mask(result_tensor, mask_tensor)
min_a = tf.placeholder(shape= [13, 13, 3, 1, 4], dtype = tf.float32)
min_b = tf.placeholder(shape = [1, 10, 4], dtype = tf.float32)
min_all = tf.minimum(min_a, min_b)
print(min_all)
#
# #     print(sess.run(result_mask, feed_dict = {result_tensor : result, mask_tensor : mask}))
# # print(class_prob * confidence)
# value = tf.constant_initializer(np.array([1,2,3]))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(value)
# print(np.array([[2],[3,4]]))



