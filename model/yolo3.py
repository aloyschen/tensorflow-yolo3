# -*- coding:utf-8 -*-
# author: gaochen
# date: 2018.06.04

import math
import numpy as np
import tensorflow as tf



def _batch_normalization_layer(input_layer, training = True, norm_decay = 0.997, norm_epsilon = 1e-5):
    '''
    Introduction
    ------------
        对卷积层提取的feature map使用batch normalization
    Parameters
    ----------
        input_layer: 输入的四维tensor
        trainging: 是否为训练过程
        norm_decay: 在预测时计算moving average时的衰减率
        norm_epsilon: 方差加上极小的数，防止除以0的情况
    Returns
    -------
        bn_layer: batch normalization处理之后的feature map
    '''
    bn_layer = tf.layers.batch_normalization(
        inputs = input_layer, axis = 3,
        momentum = norm_decay, epsilon = norm_epsilon, center = True,
        scale = True, training = training, fused = True)
    return bn_layer


def _conv2d_bn_leakyReLU(inputs, filters_num, kernel_size, strides = 1, training = True, norm_decay = 0.997, norm_epsilon = 1e-5):
    """
    Introduction
    ------------
        使用tf.layers.conv2d减少权重和偏置矩阵初始化过程，以及卷积后加上偏置项的操作
        经过卷积之后需要进行batch norm，最后使用leaky ReLU激活函数
        根据卷积时的步长，如果卷积的步长为2，则对图像进行降采样
        比如，输入图片的大小为416*416，卷积核大小为3，若stride为2时，（416 - 3 + 2）/ 2 + 1， 计算结果为208，相当于做了池化层处理
        因此需要对stride大于1的时候，先进行一个padding操作, 采用四周都padding一维代替'same'方式
    Parameters
    ----------
        inputs: 输入变量
        filters_num: 卷积核数量
        strides: 卷积步长
        trainging: 是否为训练过程
        norm_decay: 在预测时计算moving average时的衰减率
        norm_epsilon: 方差加上极小的数，防止除以0的情况
    Returns
    -------
        conv: 卷积之后的feature map
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if strides > 1:
        # 在输入feature map的长宽维度进行padding
        inputs = tf.pad(inputs, paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], mode = 'CONSTANT')
    conv = tf.layers.conv2d(
        inputs = inputs, filters = filters_num,
        kernel_size = kernel_size, strides = [strides, strides],
        padding = ('SAME' if strides == 1 else 'VALID'),
        use_bias = False, kernel_initializer = tf.variance_scaling_initializer())
    conv_bn = _batch_normalization_layer(conv, training, norm_decay, norm_epsilon)
    return tf.nn.leaky_relu(conv_bn, alpha = 0.1)


def _Residual_block(inputs, filters_num, blocks_num, training = True, norm_decay = 0.997, norm_epsilon = 1e-5):
    """
    Introduction
    ------------
        Darknet的残差block，类似resnet的两层卷积结构，分别采用1x1和3x3的卷积核，使用1x1是为了减少channel的维度
    Parameters
    ----------
        inputs: 输入变量
        filters_num: 卷积核数量
        trainging: 是否为训练过程
        blocks_num: block的数量
        norm_decay: 在预测时计算moving average时的衰减率
        norm_epsilon: 方差加上极小的数，防止除以0的情况
    Returns
    -------
        inputs: 经过残差网络处理后的结果
    """

    conv = _conv2d_bn_leakyReLU(inputs, filters_num, kernel_size = 3, strides = 2, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
    for _ in range(blocks_num):
        shortcut = conv
        conv = _conv2d_bn_leakyReLU(conv, filters_num // 2, kernel_size = 1, strides = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv = _conv2d_bn_leakyReLU(conv, filters_num, kernel_size = 3, strides = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv += shortcut
    return conv


def _darknet53(inputs, training = True, norm_decay = 0.997, norm_epsilon = 1e-5):
    """
    Introduction
    ------------
        构建yolo3使用的darknet53网络结构
    Parameters
    ----------
        inputs: 模型输入变量
    Returns
    -------
        conv: 经过52层卷积计算之后的结果, 输入图片为416x416x3，则此时输出的结果shape为13x13x1024
        route1: 返回第36层卷积计算结果52x52x256, 供后续使用
        route2: 返回第61层卷积计算结果26x26x512, 供后续使用
    """
    conv = _conv2d_bn_leakyReLU(inputs, filters_num = 32, kernel_size = 3, strides = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
    conv = _Residual_block(conv, filters_num = 64, blocks_num = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
    conv = _Residual_block(conv, filters_num = 128, blocks_num = 2, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
    conv = _Residual_block(conv, filters_num = 256, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
    route1 = conv
    conv = _Residual_block(conv, filters_num = 512, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
    route2 = conv
    conv = _Residual_block(conv, filters_num = 1024, blocks_num = 4, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
    return  route1, route2, conv


def _yolo_block(inputs, filters_num, out_filters, training = True, norm_decay = 0.997, norm_epsilon = 1e-5):
    """
    Introduction
    ------------
        yolo3在Darknet53提取的特征层基础上，又加了针对3种不同比例的feature map的block，这样来提高对小物体的检测率
    Parameters
    ----------
        inputs: 输入特征
        filters_num: 卷积核数量
        out_filters: 最后输出层的卷积核数量
    Returns
    -------
        route: 返回最后一层卷积的前一层结果
        conv: 返回最后一层卷积的结果
    """
    conv = _conv2d_bn_leakyReLU(inputs, filters_num = filters_num, kernel_size = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
    conv = _conv2d_bn_leakyReLU(conv, filters_num = filters_num * 2, kernel_size = 3, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
    conv = _conv2d_bn_leakyReLU(conv, filters_num = filters_num, kernel_size = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
    conv = _conv2d_bn_leakyReLU(conv, filters_num = filters_num * 2, kernel_size = 3, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
    conv = _conv2d_bn_leakyReLU(conv, filters_num = filters_num, kernel_size = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
    route = conv
    conv = _conv2d_bn_leakyReLU(conv, filters_num = filters_num * 2, kernel_size = 1, training = training, norm_decay = norm_decay,norm_epsilon = norm_epsilon)
    conv = tf.layers.conv2d(
        inputs = conv, filters = out_filters,
        kernel_size=1, strides=1, padding='SAME',
        use_bias=False, kernel_initializer=tf.variance_scaling_initializer())
    return route, conv


def _yolo_detection_layer(predictions, anchors, num_classes, input_shape, training = False):
    """
    Introduction
    ------------
        根据不同大小的feature map做多尺度的检测，三种feature map大小分别为13x13x1024, 26x26x512, 52x52x256
    Parameters
    ----------
        predictions: 输入的特征feature map
        anchors: 针对不同大小的feature map的anchor
        num_classes: 类别的数量
        input_shape: 图像的输入大小，一般为416
        trainging: 是否训练，用来控制返回不同的值
    Returns
    -------
    """
    num_anchors = len(anchors)
    anchors = tf.reshape(tf.constant(anchors, dtype = tf.float32), [1, 1, 1, num_anchors, 2])
    grid_size = tf.shape(predictions)[1:3]
    predictions = tf.reshape(predictions, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
    # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
    grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis = -1)
    grid = tf.cast(grid, tf.float32)
    #将x,y坐标归一化为占416的比例
    box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size, tf.float32)
    #将w,h也归一化为占416的比例
    box_wh = tf.exp(predictions[..., 2:4]) * anchors / input_shape
    box_confidence = tf.sigmoid(predictions[..., 4:5])
    box_class_probs = tf.sigmoid(predictions[..., 5:])
    if training == True:
        return grid, predictions, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_boxes_scores(feats, anchors, num_classes, input_shape, image_shape):
    """
    Introduction
    ------------
        该函数是将box的坐标修正，除去之前按照长宽比缩放填充的部分，最后将box的坐标还原成相对原始图片的
    Parameters
    ----------
        box_xy: box的中心坐标
        box_wh: box的长宽
        input_shape: 训练输入图片大小
        image_shape: 原始图片的大小
    """
    input_shape = tf.cast(input_shape, tf.float32)
    image_shape = tf.cast(image_shape, tf.float32)
    box_xy, box_wh, box_confidence, box_class_probs = _yolo_detection_layer(feats, anchors, num_classes, input_shape, training = False)
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw = box_hw * scale

    box_min = box_yx - box_hw / 2.
    box_max = box_yx + box_hw / 2.
    boxes = tf.concat(
        [box_min[..., 0:1],
         box_min[..., 1:2],
         box_max[..., 0:1],
         box_max[..., 1:2]],
        axis = -1
    )
    boxes *= tf.concat([image_shape, image_shape], axis = -1)
    boxes = tf.reshape(boxes, [-1, 4])
    boxes_scores = box_confidence * box_class_probs
    boxes_scores = tf.reshape(boxes_scores, [-1, num_classes])
    return boxes, boxes_scores


def Preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """
    Introduction
    ------------
        对训练数据的ground truth box进行预处理
    Parameters
    ----------
        true_boxes: ground truth box 形状为[batch, boxes, 5], x_min, y_min, x_max, y_max, class_id
        input_shape: 输入训练图像的长宽
        anchors: 根据数据集box聚类得到的长宽，形状为[9，2]
        num_classes: 类别数量
    """
    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    batch_image_num = true_boxes.shape[0]
    grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]
    y_true = [np.zeros((batch_image_num, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes), dtype = 'float32') for l in range(num_layers)]
    # 这里扩充维度是为了后面应用广播计算每个图中所有box的anchor互相之间的iou
    anchors = np.expand_dims(anchors, 0)
    anchors_max = anchors / 2.
    anchors_min = -anchors_max
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(batch_image_num):
        # 去除全零行数据
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # 为了应用广播扩充维度
        wh = np.expand_dims(wh, -2)
        # wh 的shape为[box_num, 1, 2]
        boxes_max = wh / 2.
        boxes_min = -boxes_max

        intersect_min = np.maximum(boxes_min, anchors_min)
        intersect_max = np.minimum(boxes_max, anchors_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        #找出和ground truth box的iou最大的anchor box, 然后将对应不同比例的负责该ground turth box 的位置置为ground truth box坐标
        best_anchor = np.argmax(iou, axis = -1)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1
    return y_true

def box_iou(box1, box2):
    """
    Introduction
    ------------
        计算box tensor之间的iou
    Parameters
    ----------
        box1: shape=[grid_size, grid_size, anchors, xywh]
        box2: shape=[box_num, xywh]
    Returns
    -------
        iou:
    """
    box1 = tf.expand_dims(box1, -2)
    box1_xy = box1[..., :2]
    box1_wh = box1[..., 2:4]
    box1_mins = box1_xy - box1_wh / 2.
    box1_maxs = box1_xy + box1_wh / 2.

    box2 = tf.expand_dims(box2, 0)
    box2_xy = box2[..., :2]
    box2_wh = box2[..., 2:4]
    box2_mins = box2_xy - box2_wh / 2.
    box2_maxs = box2_xy + box2_wh / 2.

    intersect_mins = tf.maximum(box1_mins, box2_mins)
    intersect_maxs = tf.minimum(box1_maxs, box2_maxs)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou



def yolo_loss(yolo_output, y_true, anchors, num_classes, ignore_thresh = .5):
    """
    Introduction
    ------------
        yolo模型的损失函数
    Parameters
    ----------
        yolo_output: yolo模型的输出
        y_true: 经过预处理的真实标签，shape为[batch, grid_size, grid_size, 5 + num_classes]
        anchors: yolo模型对应的anchors
        num_classes: 类别数量
        ignore_thresh: 小于该阈值的box我们认为没有物体
    Returns
    -------
        loss: 每个batch的平均损失值
    """
    loss = 0
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = [416.0, 416.0]
    grid_shapes = [tf.cast(tf.shape(yolo_output[l][1:3]), tf.float32) for l in range(3)]
    for index in range(3):
        # 只有负责预测ground truth box的grid对应的为1, 才计算相对应的loss
        # object_mask的shape为[batch_size, grid_size, grid_size, 3, 1]
        object_mask = y_true[index][..., 4:5]
        class_probs = y_true[index][..., 5:]
        grid, predictions, pred_xy, pred_wh = _yolo_detection_layer(yolo_output[index], anchors[anchor_mask[index]], num_classes, input_shape, training = True)
        # pred_box的shape为[batch
        pred_box = tf.concat([pred_xy, pred_wh], axis = -1)

        raw_true_xy = y_true[index][..., :2] * grid_shapes[index][::-1] - grid
        raw_true_wh = tf.log(tf.where(tf.equal(y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape, 0), tf.ones(
            shape = tf.shape(y_true[..., 2:4], dtype = tf.float32)), y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape))
        # 该系数是用来调整box坐标loss的系数
        box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]
        ignore_mask = tf.TensorArray(dtype = tf.float32, size = 1, dynamic_size = True)
        object_mask_bool = tf.cast(object_mask, dtype = tf.bool)
        def loop_body(index, ignore_mask):
            # true_box的shape为[box_num, 4]
            true_box = tf.boolean_mask(y_true[index][index, ..., 0:4], object_mask_bool[index, ..., 0])
            iou = box_iou(pred_box[index], true_box)
            # 计算每个true_box对应的预测的iou最大的box
            best_iou = tf.reduce_max(iou, axis = -1)
            ignore_mask = ignore_mask.write(index, tf.cast(best_iou < ignore_thresh), tf.float32)
            return index + 1, ignore_mask
        _, ignore_mask = tf.while_loop(lambda index, ignore_mask : index < tf.shape(yolo_output[0][0]), loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, axis = -1)

        # 计算四个部分的loss
        xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels = raw_true_xy, logits = predictions[..., 0:2])
        wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - predictions[..., 2:4])
        confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels = object_mask, logits = predictions[..., 4:5]) + (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels = object_mask, logits = predictions[..., 4:5]) * ignore_mask
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels =  class_probs, logits = predictions[..., 5:])
        xy_loss = tf.reduce_sum(xy_loss) / tf.cast(yolo_output[0][0], tf.float32)
        wh_loss = tf.reduce_sum(wh_loss) / tf.cast(yolo_output[0][0], tf.float32)
        confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(yolo_output[0][0], tf.float32)
        class_loss = tf.reduce_sum(class_loss) / tf.cast(yolo_output[0][0], tf.float32)
        loss += xy_loss + wh_loss + confidence_loss + class_loss
    return loss
























def yolo(inputs, training = True, norm_decay = 0.997, norm_epsilon = 1e-5):
    """
    Introduction
    ------------
        yolo模型的整体结构

    :return:
    """
    with tf.variable_scope('darknet-53'):
        route1, route2, conv = _darknet53(inputs, training, norm_decay, norm_epsilon)
    with tf.variable_scope('yolo-v3'):
        route, conv = _yolo_block(inputs, 512, training, norm_decay, norm_epsilon)





class YoloModel:
    def __init__(self):
        """
        Introduction
        ------------
            模型初始化函数
        """
    def __call__(self, inputs, training):
        """
        Introduction
        ------------
            使类的实例能够像函数一样使用
        Parameters
        ----------
            inputs: 输入变量
            training: 是否为训练过程
        Returns
        -------
            inputs: 返回模型计算结果
        """


