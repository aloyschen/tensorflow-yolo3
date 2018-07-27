import os
import config
import json
import tensorflow as tf
import numpy as np
from collections import defaultdict




class Reader:
    def __init__(self, data_dir, anchors_path, num_classes, input_shape = 416.0, max_boxes = 20, jitter = .3, hue = .1, sat = 1.5, cont = 1.5, bri = 0.2):
        """
        Introduction
        ------------
            构造函数
        Parameters
        ----------
            data_dir: 文件路径
            anchors: 数据集聚类得到的anchor
            num_classes: 数据集图片类别数量
            input_shape: 图像输入模型的大小
            max_boxes: 每张图片最大的box数量
            jitter: 随机长宽比系数
            hue: 调整hue颜色空间系数
            sat: 调整饱和度系数
            cont: 调整对比度系数
            bri: 调整亮度系数
        """
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.max_boxes = max_boxes
        self.jitter = jitter
        self.hue = hue
        self.sat = sat
        self.cont = cont
        self.bri = bri
        self.file_names = {'train' : 'train.txt', 'val' : 'val.txt'}
        self.anchors_path = anchors_path
        self.anchors = self._get_anchors()
        self.num_classes = num_classes


    def Preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
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
        input_shape = np.array([input_shape, input_shape], dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]


        grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]
        y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes), dtype='float32') for l in range(num_layers)]
        # 这里扩充维度是为了后面应用广播计算每个图中所有box的anchor互相之间的iou
        anchors = np.expand_dims(anchors, 0)
        anchors_max = anchors / 2.
        anchors_min = -anchors_max
        # 因为之前对box做了padding, 因此需要去除全0行
        valid_mask = boxes_wh[..., 0] > 0
        wh = boxes_wh[valid_mask]
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

        # 找出和ground truth box的iou最大的anchor box, 然后将对应不同比例的负责该ground turth box 的位置置为ground truth box坐标
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[t, 4].astype('int32')
                    y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                    y_true[l][j, i, k, 4] = 1
                    y_true[l][j, i, k, 5 + c] = 1
        return y_true[0], y_true[1], y_true[2]

    def _get_anchors(self):
        """
        Introduction
        ------------
            获取anchors
        Returns
        -------
            anchors: anchor数组
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


    def read_annotations(self, data_file, image_data_file):
        """
        Introduction
        ------------
            读取数据集图片路径和对应的标注
        Parameters
        ----------
            data_file: 文件路径
        """
        image_data = []
        boxes_data = []
        name_box_id = defaultdict(list)
        with open(data_file, encoding = 'utf-8') as file:
            data = json.load(file)
            annotations = data['annotations']
            for ant in annotations:
                id = ant['image_id']
                name = os.path.join(image_data_file, '%012d.jpg' % id)
                cat = ant['category_id']
                if cat >= 1 and cat <= 11:
                    cat = cat - 1
                elif cat >= 13 and cat <= 25:
                    cat = cat - 2
                elif cat >= 27 and cat <= 28:
                    cat = cat - 3
                elif cat >= 31 and cat <= 44:
                    cat = cat - 5
                elif cat >= 46 and cat <= 65:
                    cat = cat - 6
                elif cat == 67:
                    cat = cat - 7
                elif cat == 70:
                    cat = cat - 9
                elif cat >= 72 and cat <= 82:
                    cat = cat - 10
                elif cat >= 84 and cat <= 90:
                    cat = cat - 11

                name_box_id[name].append([ant['bbox'], cat])

            for key in name_box_id.keys():
                boxes = []
                image_data.append(key)
                box_infos = name_box_id[key]
                for info in box_infos:
                    x_min = info[0][0]
                    y_min = info[0][1]
                    x_max = x_min + info[0][2]
                    y_max = y_min + info[0][3]
                    boxes.append(np.array([x_min, y_min, x_max, y_max, info[1]]))
                boxes_data.append(np.array(boxes))
        return image_data, boxes_data


    def convert_to_tfrecord(self, mode, tfrecord_path, annotations_path, image_data_file):
        """
        Introduction
        ------------
            将图片和boxes数据存储为tfRecord
        Parameters
        ----------
            mode: 训练集还是验证集的数据
            tfrecord_path: tfrecord文件存储路径
            annotations_path: 标注文件的路径
            image_data_file: 图片文件路径
            path: tfRecord的路径
        """
        output_file = os.path.join(tfrecord_path, mode + '.tfrecords')
        image_data, boxes_data = self.read_annotations(annotations_path, image_data_file)
        with tf.python_io.TFRecordWriter(output_file) as record_writer:
            for index in range(len(image_data)):
                with tf.gfile.FastGFile(image_data[index], 'rb') as file:
                    image = file.read()
                    xmin, xmax, ymin, ymax, label = [], [], [], [], []
                    for box in boxes_data[index]:
                        xmin.append(box[0])
                        ymin.append(box[1])
                        xmax.append(box[2])
                        ymax.append(box[3])
                        label.append(box[4])
                    example = tf.train.Example(features = tf.train.Features(
                        feature = {
                            'image/encoded' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                            'image/object/bbox/xmin' : tf.train.Feature(float_list = tf.train.FloatList(value = xmin)),
                            'image/object/bbox/xmax': tf.train.Feature(float_list = tf.train.FloatList(value = xmax)),
                            'image/object/bbox/ymin': tf.train.Feature(float_list = tf.train.FloatList(value = ymin)),
                            'image/object/bbox/ymax': tf.train.Feature(float_list = tf.train.FloatList(value = ymax)),
                            'image/object/bbox/label': tf.train.Feature(float_list = tf.train.FloatList(value = label)),
                        }
                    ))
                    record_writer.write(example.SerializeToString())
                    if index % 1000 == 0:
                        print('Processed {} of {} images'.format(index + 1, len(image_data)))


    def parser(self, serialized_example):
        """
        Introduction
        ------------
            对所有的数据集每条数据进行处理
        Parameters
        ----------
            image_file: 图片路径
            box: 图片对应的box坐标
        """
        features = tf.parse_single_example(
            serialized_example,
            features = {
                'image/encoded' : tf.FixedLenFeature([], dtype = tf.string),
                'image/object/bbox/xmin' : tf.VarLenFeature(dtype = tf.float32),
                'image/object/bbox/xmax': tf.VarLenFeature(dtype = tf.float32),
                'image/object/bbox/ymin': tf.VarLenFeature(dtype = tf.float32),
                'image/object/bbox/ymax': tf.VarLenFeature(dtype = tf.float32),
                'image/object/bbox/label': tf.VarLenFeature(dtype = tf.float32)
            }
        )
        image = tf.image.decode_jpeg(features['image/encoded'], channels = 3)
        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, axis = 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, axis = 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, axis = 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, axis = 0)
        label = tf.expand_dims(features['image/object/bbox/label'].values, axis = 0)
        bbox = tf.concat(axis = 0, values = [xmin, ymin, xmax, ymax, label])
        bbox = tf.transpose(bbox, [1, 0])
        image, bbox_true_13, bbox_true_26, bbox_true_52 = self.Preprocess(image, bbox)
        return image, bbox_true_13, bbox_true_26, bbox_true_52



    def Preprocess(self, image, bbox):
        """
        Introduction
        ------------
            对图片进行预处理，增强数据集
        Parameters
        ----------
            image: tensorflow解析的图片
            bbox: 图片中对应的box坐标
        """
        image_width, image_high = tf.shape(image)[1], tf.shape(image)[0]
        input_width = self.input_shape
        input_high = self.input_shape
        # 随机长宽比
        new_aspect_ratio = tf.random_uniform([], dtype = tf.float32, minval = 1 - self.jitter, maxval = 1 + self.jitter)
        # 图片随机缩放比例
        scale = tf.random_uniform([], dtype = tf.float32, minval = .25, maxval = 2)
        new_high, new_width = tf.cond(tf.less(new_aspect_ratio, 1), lambda : (tf.cast(scale * input_high, tf.int32), tf.cast(scale * input_high * new_aspect_ratio, tf.int32)), lambda : (tf.cast(scale * input_width / new_aspect_ratio, tf.int32), tf.cast(scale * input_width, tf.int32)))
        image = tf.image.resize_images(image, [new_high, new_width], align_corners = True)

        # 将图片按照裁剪或者padding的策略缩放
        dx = tf.divide(tf.subtract(input_width, tf.cast(new_width, tf.float32)), 2)
        dy = tf.divide(tf.subtract(input_high, tf.cast(new_high, tf.float32)), 2)
        image = tf.image.resize_image_with_crop_or_pad(image, tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))

        # 随机左右翻转图片
        flip_left_right = tf.greater(tf.random_uniform([], dtype = tf.float32, minval = 0, maxval = 1), 0.5)
        image = tf.cond(flip_left_right, lambda : tf.image.flip_left_right(image), lambda : image)

        # 随机上下翻转图片
        flip_up_down = tf.greater(tf.random_uniform([], dtype = tf.float32, minval = 0, maxval = 1), 0.5)
        image = tf.cond(flip_up_down, lambda : tf.image.flip_up_down(image), lambda : image)

        # 随机调整颜色
        delta = tf.random_uniform([], dtype = tf.float32, minval = -self.hue, maxval = self.hue)
        image = tf.image.adjust_hue(image / 255, delta) * 255
        image = tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 255.0)

        # 随机调整饱和度
        saturation_factor = tf.random_uniform([], dtype = tf.float32, minval = 1, maxval = self.sat)
        image = tf.image.adjust_saturation(image / 255, saturation_factor) * 255
        image = tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 255.0)

        # 随机调整对比度
        contrast_factor = tf.random_uniform([], dtype = tf.float32, minval = 1, maxval = self.cont)
        image = tf.image.adjust_contrast(image / 255, contrast_factor) * 255
        image = tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 255.0)

        # 随机调整亮度
        bright_factor = tf.random_uniform([], dtype = tf.float32, minval = -self.bri, maxval = self.bri)
        image = tf.image.adjust_brightness(image / 255, bright_factor) * 255
        image = tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 255.0)

        # 将图片归一化到0和1之间
        image = image / 255
        image = tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)
        # 矫正box坐标
        def _flip_left_right_boxes(boxes):
            xmin, ymin, xmax, ymax, label = tf.split(value = boxes, num_or_size_splits = 5, axis = 1)
            flipped_xmin = tf.subtract(input_width, xmax)
            flipped_xmax = tf.subtract(input_width, xmin)
            flipped_boxes = tf.concat([flipped_xmin, ymin, flipped_xmax, ymax, label], 1)
            return flipped_boxes

        def _flip_up_down_boxes(boxes):
            xmin, ymin, xmax, ymax, label = tf.split(value = boxes, num_or_size_splits = 5, axis = 1)
            flipped_ymin = tf.subtract(input_high, ymax)
            flipped_ymax = tf.subtract(input_high, ymin)
            flipped_boxes = tf.concat([xmin, flipped_ymin, xmax, flipped_ymax, label], 1)
            return flipped_boxes

        def _resize_boxes(boxes):
            xmin, ymin, xmax, ymax, label = tf.split(value=boxes, num_or_size_splits = 5, axis = 1)
            xmin = xmin * tf.cast(new_width / image_width, tf.float32) + tf.cast(dx, tf.float32)
            xmax = xmax * tf.cast(new_width / image_width, tf.float32) + tf.cast(dx, tf.float32)
            ymin = ymin * tf.cast(new_high / image_high, tf.float32) + tf.cast(dy, tf.float32)
            ymax = ymax * tf.cast(new_high / image_high, tf.float32) + tf.cast(dy, tf.float32)
            boxes = tf.concat([xmin, ymin, xmax, ymax, label], 1)
            return boxes
        bbox = _resize_boxes(bbox)
        bbox = tf.cond(flip_left_right, lambda: _flip_left_right_boxes(bbox), lambda: bbox)
        bbox = tf.cond(flip_up_down, lambda: _flip_up_down_boxes(bbox), lambda: bbox)
        bbox = tf.clip_by_value(bbox, clip_value_min=0, clip_value_max = input_width)
        bbox = tf.cond(tf.greater(tf.shape(bbox)[0], 20), lambda: bbox[:20], lambda: tf.pad(bbox, paddings = [[0, 20 - tf.shape(bbox)[0]], [0, 0]], mode = 'CONSTANT'))
        bbox_true_13, bbox_true_26, bbox_true_52 = tf.py_func(self.Preprocess_true_boxes, [bbox, self.anchors, self.input_shape, self.num_classes], [tf.float32, tf.float32, tf.float32])
        return image, bbox_true_13, bbox_true_26, bbox_true_52

    def make_batch(self, mode, batch_size, annotations_file, data_file):
        """
        Introduction
        ------------
            读取训练集、验证集、测试集的tfRecord的数据
        Parameters
        ----------
            mode: 标识训练集、验证集和测试集
            batch_size: batch的大小
            annotations_file: 标注文件
            data_file: 数据集路径
        Returns
        -------
        """
        TfrecordFile = os.path.join(self.data_dir, mode + '.tfrecords')
        if not os.path.exists(TfrecordFile):
            self.convert_to_tfrecord(mode, self.data_dir, annotations_file, data_file)
        Dataset = tf.data.TFRecordDataset(filenames = TfrecordFile).repeat()
        Dataset = Dataset.map(self.parser, num_parallel_calls = config.num_parallel_calls)
        if mode == 'train':
            Dataset = Dataset.shuffle(buffer_size = config.train_num).repeat()
        # 并行的对数据进行map预处理
        Dataset = Dataset.map(lambda image_files, boxes: tf.py_func(self.parser, [image_files, boxes], [tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls = config.num_parallel_calls)
        Dataset = Dataset.batch(batch_size)
        iterator = Dataset.make_one_shot_iterator()
        image_batch, bbox_true_13, bbox_true_26, bbox_true_52 = iterator.get_next()
        # 这里因为tf.pyfunc返回的tensor没有shape, 所以需要重新设置一下
        grid_shapes = [self.input_shape // 32, self.input_shape // 16, self.input_shape // 8]
        image_batch.set_shape([batch_size, self.input_shape, self.input_shape, 3])
        bbox_true_13.set_shape([batch_size, grid_shapes[0], grid_shapes[0], 3, 5 + self.num_classes])
        bbox_true_26.set_shape([batch_size, grid_shapes[1], grid_shapes[1], 3, 5 + self.num_classes])
        bbox_true_52.set_shape([batch_size, grid_shapes[2], grid_shapes[2], 3, 5 + self.num_classes])
        return image_batch, bbox_true_13, bbox_true_26, bbox_true_52

