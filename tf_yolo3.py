import tensorflow as tf

__weights_dict = dict()

is_train = False

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    input_1         = tf.placeholder(tf.float32,  shape = (None, 416, 416, 3), name = 'input_1')
    conv2d_1        = convolution(input_1, group=1, strides=[1, 1], padding='SAME', name='conv2d_1')
    batch_normalization_1 = batch_normalization(conv2d_1, variance_epsilon=0.0010000000474974513, name='batch_normalization_1')
    leaky_re_lu_1   = tf.nn.leaky_relu(batch_normalization_1, alpha=0.10000000149011612, name='leaky_re_lu_1')
    zero_padding2d_1 = tf.pad(leaky_re_lu_1, [[0, 0], [1, 0], [1, 0], [0, 0]], 'CONSTANT', name='zero_padding2d_1')
    conv2d_2        = convolution(zero_padding2d_1, group=1, strides=[2, 2], padding='VALID', name='conv2d_2')
    batch_normalization_2 = batch_normalization(conv2d_2, variance_epsilon=0.0010000000474974513, name='batch_normalization_2')
    leaky_re_lu_2   = tf.nn.leaky_relu(batch_normalization_2, alpha=0.10000000149011612, name='leaky_re_lu_2')
    conv2d_3        = convolution(leaky_re_lu_2, group=1, strides=[1, 1], padding='SAME', name='conv2d_3')
    batch_normalization_3 = batch_normalization(conv2d_3, variance_epsilon=0.0010000000474974513, name='batch_normalization_3')
    leaky_re_lu_3   = tf.nn.leaky_relu(batch_normalization_3, alpha=0.10000000149011612, name='leaky_re_lu_3')
    conv2d_4        = convolution(leaky_re_lu_3, group=1, strides=[1, 1], padding='SAME', name='conv2d_4')
    batch_normalization_4 = batch_normalization(conv2d_4, variance_epsilon=0.0010000000474974513, name='batch_normalization_4')
    leaky_re_lu_4   = tf.nn.leaky_relu(batch_normalization_4, alpha=0.10000000149011612, name='leaky_re_lu_4')
    add_1           = leaky_re_lu_2 + leaky_re_lu_4
    zero_padding2d_2 = tf.pad(add_1, [[0, 0], [1, 0], [1, 0], [0, 0]], 'CONSTANT', name='zero_padding2d_2')
    conv2d_5        = convolution(zero_padding2d_2, group=1, strides=[2, 2], padding='VALID', name='conv2d_5')
    batch_normalization_5 = batch_normalization(conv2d_5, variance_epsilon=0.0010000000474974513, name='batch_normalization_5')
    leaky_re_lu_5   = tf.nn.leaky_relu(batch_normalization_5, alpha=0.10000000149011612, name='leaky_re_lu_5')
    conv2d_6        = convolution(leaky_re_lu_5, group=1, strides=[1, 1], padding='SAME', name='conv2d_6')
    batch_normalization_6 = batch_normalization(conv2d_6, variance_epsilon=0.0010000000474974513, name='batch_normalization_6')
    leaky_re_lu_6   = tf.nn.leaky_relu(batch_normalization_6, alpha=0.10000000149011612, name='leaky_re_lu_6')
    conv2d_7        = convolution(leaky_re_lu_6, group=1, strides=[1, 1], padding='SAME', name='conv2d_7')
    batch_normalization_7 = batch_normalization(conv2d_7, variance_epsilon=0.0010000000474974513, name='batch_normalization_7')
    leaky_re_lu_7   = tf.nn.leaky_relu(batch_normalization_7, alpha=0.10000000149011612, name='leaky_re_lu_7')
    add_2           = leaky_re_lu_5 + leaky_re_lu_7
    conv2d_8        = convolution(add_2, group=1, strides=[1, 1], padding='SAME', name='conv2d_8')
    batch_normalization_8 = batch_normalization(conv2d_8, variance_epsilon=0.0010000000474974513, name='batch_normalization_8')
    leaky_re_lu_8   = tf.nn.leaky_relu(batch_normalization_8, alpha=0.10000000149011612, name='leaky_re_lu_8')
    conv2d_9        = convolution(leaky_re_lu_8, group=1, strides=[1, 1], padding='SAME', name='conv2d_9')
    batch_normalization_9 = batch_normalization(conv2d_9, variance_epsilon=0.0010000000474974513, name='batch_normalization_9')
    leaky_re_lu_9   = tf.nn.leaky_relu(batch_normalization_9, alpha=0.10000000149011612, name='leaky_re_lu_9')
    add_3           = add_2 + leaky_re_lu_9
    zero_padding2d_3 = tf.pad(add_3, [[0, 0], [1, 0], [1, 0], [0, 0]], 'CONSTANT', name='zero_padding2d_3')
    conv2d_10       = convolution(zero_padding2d_3, group=1, strides=[2, 2], padding='VALID', name='conv2d_10')
    batch_normalization_10 = batch_normalization(conv2d_10, variance_epsilon=0.0010000000474974513, name='batch_normalization_10')
    leaky_re_lu_10  = tf.nn.leaky_relu(batch_normalization_10, alpha=0.10000000149011612, name='leaky_re_lu_10')
    conv2d_11       = convolution(leaky_re_lu_10, group=1, strides=[1, 1], padding='SAME', name='conv2d_11')
    batch_normalization_11 = batch_normalization(conv2d_11, variance_epsilon=0.0010000000474974513, name='batch_normalization_11')
    leaky_re_lu_11  = tf.nn.leaky_relu(batch_normalization_11, alpha=0.10000000149011612, name='leaky_re_lu_11')
    conv2d_12       = convolution(leaky_re_lu_11, group=1, strides=[1, 1], padding='SAME', name='conv2d_12')
    batch_normalization_12 = batch_normalization(conv2d_12, variance_epsilon=0.0010000000474974513, name='batch_normalization_12')
    leaky_re_lu_12  = tf.nn.leaky_relu(batch_normalization_12, alpha=0.10000000149011612, name='leaky_re_lu_12')
    add_4           = leaky_re_lu_10 + leaky_re_lu_12
    conv2d_13       = convolution(add_4, group=1, strides=[1, 1], padding='SAME', name='conv2d_13')
    batch_normalization_13 = batch_normalization(conv2d_13, variance_epsilon=0.0010000000474974513, name='batch_normalization_13')
    leaky_re_lu_13  = tf.nn.leaky_relu(batch_normalization_13, alpha=0.10000000149011612, name='leaky_re_lu_13')
    conv2d_14       = convolution(leaky_re_lu_13, group=1, strides=[1, 1], padding='SAME', name='conv2d_14')
    batch_normalization_14 = batch_normalization(conv2d_14, variance_epsilon=0.0010000000474974513, name='batch_normalization_14')
    leaky_re_lu_14  = tf.nn.leaky_relu(batch_normalization_14, alpha=0.10000000149011612, name='leaky_re_lu_14')
    add_5           = add_4 + leaky_re_lu_14
    conv2d_15       = convolution(add_5, group=1, strides=[1, 1], padding='SAME', name='conv2d_15')
    batch_normalization_15 = batch_normalization(conv2d_15, variance_epsilon=0.0010000000474974513, name='batch_normalization_15')
    leaky_re_lu_15  = tf.nn.leaky_relu(batch_normalization_15, alpha=0.10000000149011612, name='leaky_re_lu_15')
    conv2d_16       = convolution(leaky_re_lu_15, group=1, strides=[1, 1], padding='SAME', name='conv2d_16')
    batch_normalization_16 = batch_normalization(conv2d_16, variance_epsilon=0.0010000000474974513, name='batch_normalization_16')
    leaky_re_lu_16  = tf.nn.leaky_relu(batch_normalization_16, alpha=0.10000000149011612, name='leaky_re_lu_16')
    add_6           = add_5 + leaky_re_lu_16
    conv2d_17       = convolution(add_6, group=1, strides=[1, 1], padding='SAME', name='conv2d_17')
    batch_normalization_17 = batch_normalization(conv2d_17, variance_epsilon=0.0010000000474974513, name='batch_normalization_17')
    leaky_re_lu_17  = tf.nn.leaky_relu(batch_normalization_17, alpha=0.10000000149011612, name='leaky_re_lu_17')
    conv2d_18       = convolution(leaky_re_lu_17, group=1, strides=[1, 1], padding='SAME', name='conv2d_18')
    batch_normalization_18 = batch_normalization(conv2d_18, variance_epsilon=0.0010000000474974513, name='batch_normalization_18')
    leaky_re_lu_18  = tf.nn.leaky_relu(batch_normalization_18, alpha=0.10000000149011612, name='leaky_re_lu_18')
    add_7           = add_6 + leaky_re_lu_18
    conv2d_19       = convolution(add_7, group=1, strides=[1, 1], padding='SAME', name='conv2d_19')
    batch_normalization_19 = batch_normalization(conv2d_19, variance_epsilon=0.0010000000474974513, name='batch_normalization_19')
    leaky_re_lu_19  = tf.nn.leaky_relu(batch_normalization_19, alpha=0.10000000149011612, name='leaky_re_lu_19')
    conv2d_20       = convolution(leaky_re_lu_19, group=1, strides=[1, 1], padding='SAME', name='conv2d_20')
    batch_normalization_20 = batch_normalization(conv2d_20, variance_epsilon=0.0010000000474974513, name='batch_normalization_20')
    leaky_re_lu_20  = tf.nn.leaky_relu(batch_normalization_20, alpha=0.10000000149011612, name='leaky_re_lu_20')
    add_8           = add_7 + leaky_re_lu_20
    conv2d_21       = convolution(add_8, group=1, strides=[1, 1], padding='SAME', name='conv2d_21')
    batch_normalization_21 = batch_normalization(conv2d_21, variance_epsilon=0.0010000000474974513, name='batch_normalization_21')
    leaky_re_lu_21  = tf.nn.leaky_relu(batch_normalization_21, alpha=0.10000000149011612, name='leaky_re_lu_21')
    conv2d_22       = convolution(leaky_re_lu_21, group=1, strides=[1, 1], padding='SAME', name='conv2d_22')
    batch_normalization_22 = batch_normalization(conv2d_22, variance_epsilon=0.0010000000474974513, name='batch_normalization_22')
    leaky_re_lu_22  = tf.nn.leaky_relu(batch_normalization_22, alpha=0.10000000149011612, name='leaky_re_lu_22')
    add_9           = add_8 + leaky_re_lu_22
    conv2d_23       = convolution(add_9, group=1, strides=[1, 1], padding='SAME', name='conv2d_23')
    batch_normalization_23 = batch_normalization(conv2d_23, variance_epsilon=0.0010000000474974513, name='batch_normalization_23')
    leaky_re_lu_23  = tf.nn.leaky_relu(batch_normalization_23, alpha=0.10000000149011612, name='leaky_re_lu_23')
    conv2d_24       = convolution(leaky_re_lu_23, group=1, strides=[1, 1], padding='SAME', name='conv2d_24')
    batch_normalization_24 = batch_normalization(conv2d_24, variance_epsilon=0.0010000000474974513, name='batch_normalization_24')
    leaky_re_lu_24  = tf.nn.leaky_relu(batch_normalization_24, alpha=0.10000000149011612, name='leaky_re_lu_24')
    add_10          = add_9 + leaky_re_lu_24
    conv2d_25       = convolution(add_10, group=1, strides=[1, 1], padding='SAME', name='conv2d_25')
    batch_normalization_25 = batch_normalization(conv2d_25, variance_epsilon=0.0010000000474974513, name='batch_normalization_25')
    leaky_re_lu_25  = tf.nn.leaky_relu(batch_normalization_25, alpha=0.10000000149011612, name='leaky_re_lu_25')
    conv2d_26       = convolution(leaky_re_lu_25, group=1, strides=[1, 1], padding='SAME', name='conv2d_26')
    batch_normalization_26 = batch_normalization(conv2d_26, variance_epsilon=0.0010000000474974513, name='batch_normalization_26')
    leaky_re_lu_26  = tf.nn.leaky_relu(batch_normalization_26, alpha=0.10000000149011612, name='leaky_re_lu_26')
    add_11          = add_10 + leaky_re_lu_26
    zero_padding2d_4 = tf.pad(add_11, [[0, 0], [1, 0], [1, 0], [0, 0]], 'CONSTANT', name='zero_padding2d_4')
    conv2d_27       = convolution(zero_padding2d_4, group=1, strides=[2, 2], padding='VALID', name='conv2d_27')
    batch_normalization_27 = batch_normalization(conv2d_27, variance_epsilon=0.0010000000474974513, name='batch_normalization_27')
    leaky_re_lu_27  = tf.nn.leaky_relu(batch_normalization_27, alpha=0.10000000149011612, name='leaky_re_lu_27')
    conv2d_28       = convolution(leaky_re_lu_27, group=1, strides=[1, 1], padding='SAME', name='conv2d_28')
    batch_normalization_28 = batch_normalization(conv2d_28, variance_epsilon=0.0010000000474974513, name='batch_normalization_28')
    leaky_re_lu_28  = tf.nn.leaky_relu(batch_normalization_28, alpha=0.10000000149011612, name='leaky_re_lu_28')
    conv2d_29       = convolution(leaky_re_lu_28, group=1, strides=[1, 1], padding='SAME', name='conv2d_29')
    batch_normalization_29 = batch_normalization(conv2d_29, variance_epsilon=0.0010000000474974513, name='batch_normalization_29')
    leaky_re_lu_29  = tf.nn.leaky_relu(batch_normalization_29, alpha=0.10000000149011612, name='leaky_re_lu_29')
    add_12          = leaky_re_lu_27 + leaky_re_lu_29
    conv2d_30       = convolution(add_12, group=1, strides=[1, 1], padding='SAME', name='conv2d_30')
    batch_normalization_30 = batch_normalization(conv2d_30, variance_epsilon=0.0010000000474974513, name='batch_normalization_30')
    leaky_re_lu_30  = tf.nn.leaky_relu(batch_normalization_30, alpha=0.10000000149011612, name='leaky_re_lu_30')
    conv2d_31       = convolution(leaky_re_lu_30, group=1, strides=[1, 1], padding='SAME', name='conv2d_31')
    batch_normalization_31 = batch_normalization(conv2d_31, variance_epsilon=0.0010000000474974513, name='batch_normalization_31')
    leaky_re_lu_31  = tf.nn.leaky_relu(batch_normalization_31, alpha=0.10000000149011612, name='leaky_re_lu_31')
    add_13          = add_12 + leaky_re_lu_31
    conv2d_32       = convolution(add_13, group=1, strides=[1, 1], padding='SAME', name='conv2d_32')
    batch_normalization_32 = batch_normalization(conv2d_32, variance_epsilon=0.0010000000474974513, name='batch_normalization_32')
    leaky_re_lu_32  = tf.nn.leaky_relu(batch_normalization_32, alpha=0.10000000149011612, name='leaky_re_lu_32')
    conv2d_33       = convolution(leaky_re_lu_32, group=1, strides=[1, 1], padding='SAME', name='conv2d_33')
    batch_normalization_33 = batch_normalization(conv2d_33, variance_epsilon=0.0010000000474974513, name='batch_normalization_33')
    leaky_re_lu_33  = tf.nn.leaky_relu(batch_normalization_33, alpha=0.10000000149011612, name='leaky_re_lu_33')
    add_14          = add_13 + leaky_re_lu_33
    conv2d_34       = convolution(add_14, group=1, strides=[1, 1], padding='SAME', name='conv2d_34')
    batch_normalization_34 = batch_normalization(conv2d_34, variance_epsilon=0.0010000000474974513, name='batch_normalization_34')
    leaky_re_lu_34  = tf.nn.leaky_relu(batch_normalization_34, alpha=0.10000000149011612, name='leaky_re_lu_34')
    conv2d_35       = convolution(leaky_re_lu_34, group=1, strides=[1, 1], padding='SAME', name='conv2d_35')
    batch_normalization_35 = batch_normalization(conv2d_35, variance_epsilon=0.0010000000474974513, name='batch_normalization_35')
    leaky_re_lu_35  = tf.nn.leaky_relu(batch_normalization_35, alpha=0.10000000149011612, name='leaky_re_lu_35')
    add_15          = add_14 + leaky_re_lu_35
    conv2d_36       = convolution(add_15, group=1, strides=[1, 1], padding='SAME', name='conv2d_36')
    batch_normalization_36 = batch_normalization(conv2d_36, variance_epsilon=0.0010000000474974513, name='batch_normalization_36')
    leaky_re_lu_36  = tf.nn.leaky_relu(batch_normalization_36, alpha=0.10000000149011612, name='leaky_re_lu_36')
    conv2d_37       = convolution(leaky_re_lu_36, group=1, strides=[1, 1], padding='SAME', name='conv2d_37')
    batch_normalization_37 = batch_normalization(conv2d_37, variance_epsilon=0.0010000000474974513, name='batch_normalization_37')
    leaky_re_lu_37  = tf.nn.leaky_relu(batch_normalization_37, alpha=0.10000000149011612, name='leaky_re_lu_37')
    add_16          = add_15 + leaky_re_lu_37
    conv2d_38       = convolution(add_16, group=1, strides=[1, 1], padding='SAME', name='conv2d_38')
    batch_normalization_38 = batch_normalization(conv2d_38, variance_epsilon=0.0010000000474974513, name='batch_normalization_38')
    leaky_re_lu_38  = tf.nn.leaky_relu(batch_normalization_38, alpha=0.10000000149011612, name='leaky_re_lu_38')
    conv2d_39       = convolution(leaky_re_lu_38, group=1, strides=[1, 1], padding='SAME', name='conv2d_39')
    batch_normalization_39 = batch_normalization(conv2d_39, variance_epsilon=0.0010000000474974513, name='batch_normalization_39')
    leaky_re_lu_39  = tf.nn.leaky_relu(batch_normalization_39, alpha=0.10000000149011612, name='leaky_re_lu_39')
    add_17          = add_16 + leaky_re_lu_39
    conv2d_40       = convolution(add_17, group=1, strides=[1, 1], padding='SAME', name='conv2d_40')
    batch_normalization_40 = batch_normalization(conv2d_40, variance_epsilon=0.0010000000474974513, name='batch_normalization_40')
    leaky_re_lu_40  = tf.nn.leaky_relu(batch_normalization_40, alpha=0.10000000149011612, name='leaky_re_lu_40')
    conv2d_41       = convolution(leaky_re_lu_40, group=1, strides=[1, 1], padding='SAME', name='conv2d_41')
    batch_normalization_41 = batch_normalization(conv2d_41, variance_epsilon=0.0010000000474974513, name='batch_normalization_41')
    leaky_re_lu_41  = tf.nn.leaky_relu(batch_normalization_41, alpha=0.10000000149011612, name='leaky_re_lu_41')
    add_18          = add_17 + leaky_re_lu_41
    conv2d_42       = convolution(add_18, group=1, strides=[1, 1], padding='SAME', name='conv2d_42')
    batch_normalization_42 = batch_normalization(conv2d_42, variance_epsilon=0.0010000000474974513, name='batch_normalization_42')
    leaky_re_lu_42  = tf.nn.leaky_relu(batch_normalization_42, alpha=0.10000000149011612, name='leaky_re_lu_42')
    conv2d_43       = convolution(leaky_re_lu_42, group=1, strides=[1, 1], padding='SAME', name='conv2d_43')
    batch_normalization_43 = batch_normalization(conv2d_43, variance_epsilon=0.0010000000474974513, name='batch_normalization_43')
    leaky_re_lu_43  = tf.nn.leaky_relu(batch_normalization_43, alpha=0.10000000149011612, name='leaky_re_lu_43')
    add_19          = add_18 + leaky_re_lu_43
    zero_padding2d_5 = tf.pad(add_19, [[0, 0], [1, 0], [1, 0], [0, 0]], 'CONSTANT', name='zero_padding2d_5')
    conv2d_44       = convolution(zero_padding2d_5, group=1, strides=[2, 2], padding='VALID', name='conv2d_44')
    batch_normalization_44 = batch_normalization(conv2d_44, variance_epsilon=0.0010000000474974513, name='batch_normalization_44')
    leaky_re_lu_44  = tf.nn.leaky_relu(batch_normalization_44, alpha=0.10000000149011612, name='leaky_re_lu_44')
    conv2d_45       = convolution(leaky_re_lu_44, group=1, strides=[1, 1], padding='SAME', name='conv2d_45')
    batch_normalization_45 = batch_normalization(conv2d_45, variance_epsilon=0.0010000000474974513, name='batch_normalization_45')
    leaky_re_lu_45  = tf.nn.leaky_relu(batch_normalization_45, alpha=0.10000000149011612, name='leaky_re_lu_45')
    conv2d_46       = convolution(leaky_re_lu_45, group=1, strides=[1, 1], padding='SAME', name='conv2d_46')
    batch_normalization_46 = batch_normalization(conv2d_46, variance_epsilon=0.0010000000474974513, name='batch_normalization_46')
    leaky_re_lu_46  = tf.nn.leaky_relu(batch_normalization_46, alpha=0.10000000149011612, name='leaky_re_lu_46')
    add_20          = leaky_re_lu_44 + leaky_re_lu_46
    conv2d_47       = convolution(add_20, group=1, strides=[1, 1], padding='SAME', name='conv2d_47')
    batch_normalization_47 = batch_normalization(conv2d_47, variance_epsilon=0.0010000000474974513, name='batch_normalization_47')
    leaky_re_lu_47  = tf.nn.leaky_relu(batch_normalization_47, alpha=0.10000000149011612, name='leaky_re_lu_47')
    conv2d_48       = convolution(leaky_re_lu_47, group=1, strides=[1, 1], padding='SAME', name='conv2d_48')
    batch_normalization_48 = batch_normalization(conv2d_48, variance_epsilon=0.0010000000474974513, name='batch_normalization_48')
    leaky_re_lu_48  = tf.nn.leaky_relu(batch_normalization_48, alpha=0.10000000149011612, name='leaky_re_lu_48')
    add_21          = add_20 + leaky_re_lu_48
    conv2d_49       = convolution(add_21, group=1, strides=[1, 1], padding='SAME', name='conv2d_49')
    batch_normalization_49 = batch_normalization(conv2d_49, variance_epsilon=0.0010000000474974513, name='batch_normalization_49')
    leaky_re_lu_49  = tf.nn.leaky_relu(batch_normalization_49, alpha=0.10000000149011612, name='leaky_re_lu_49')
    conv2d_50       = convolution(leaky_re_lu_49, group=1, strides=[1, 1], padding='SAME', name='conv2d_50')
    batch_normalization_50 = batch_normalization(conv2d_50, variance_epsilon=0.0010000000474974513, name='batch_normalization_50')
    leaky_re_lu_50  = tf.nn.leaky_relu(batch_normalization_50, alpha=0.10000000149011612, name='leaky_re_lu_50')
    add_22          = add_21 + leaky_re_lu_50
    conv2d_51       = convolution(add_22, group=1, strides=[1, 1], padding='SAME', name='conv2d_51')
    batch_normalization_51 = batch_normalization(conv2d_51, variance_epsilon=0.0010000000474974513, name='batch_normalization_51')
    leaky_re_lu_51  = tf.nn.leaky_relu(batch_normalization_51, alpha=0.10000000149011612, name='leaky_re_lu_51')
    conv2d_52       = convolution(leaky_re_lu_51, group=1, strides=[1, 1], padding='SAME', name='conv2d_52')
    batch_normalization_52 = batch_normalization(conv2d_52, variance_epsilon=0.0010000000474974513, name='batch_normalization_52')
    leaky_re_lu_52  = tf.nn.leaky_relu(batch_normalization_52, alpha=0.10000000149011612, name='leaky_re_lu_52')
    add_23          = add_22 + leaky_re_lu_52
    return input_1, add_23


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer

def batch_normalization(input, name, **kwargs):
    mean = tf.Variable(__weights_dict[name]['mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(__weights_dict[name]['var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(__weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in __weights_dict[name] else None
    scale = tf.Variable(__weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in __weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name = name, **kwargs)

