import os
import config
import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from model.yolo3 import Preprocess_true_boxes

class dataReader:
    def __init__(self, data_dir, anchors, num_classes, input_shape = 416, max_boxes = 20, jitter = .3, hue = .1, sat = 1.5, val = 1.5):
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
            hue: H通道数比例
            sat: S通道数比例
            val: V通道数比例
        """
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.max_boxes = max_boxes
        self.jitter = jitter
        self.hue = hue
        self.sat = sat
        self.val = val
        self.file_names = {'train' : '2012_train.txt', 'val' : '2012_val.txt'}
        self.anchors = anchors
        self.num_classes = num_classes



    def read_annotations(self, data_file):
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
        with open(data_file, encoding = 'utf-8') as file:
            data = file.readlines()
            for line in data:
                line = line.split()
                image_file = line[0]
                box = np.array([np.array(list(box.split(",")), np.int32) for box in line[1:]])
                image, box = self.get_random_data(image_file, box)
                image_data.append(image)
                boxes_data.append(box)
        return image_data, boxes_data



    def get_random_data(self, image_file, box):
        """
        Introduction
        ------------
            对图像训练数据集进行随机处理，来增强泛化能力
        Parameters
        ----------
            image_file: 图像的路径
            box: 图像当中box的标注
        Returns
        -------
            image_data: 图像数据
            box_data: 标注数据
        """
        image = Image.open(image_file)
        image_width, image_high = image.size
        input_width, input_high = self.input_shape
        # 随机长宽比
        new_aspect_ratio = np.random.uniform(1 - self.jitter,1 + self.jitter)/np.random.uniform(1 - self.jitter, 1 + self.jitter)
        # 图片随机缩放比例
        scale = np.random.uniform(.25, 2)
        if new_aspect_ratio < 1:
            new_high = int(scale * input_high)
            new_width = int(new_high * new_aspect_ratio)
        else:
            new_width = int(scale * input_width)
            new_high = int(new_width / new_aspect_ratio)
        image = image.resize((new_width, new_high), Image.BICUBIC)

        # 将图片随机粘贴在416，416的图片上
        dx = int(np.random.uniform(0, input_width - new_width))
        dy = int(np.random.uniform(0, input_high - new_high))
        new_image = Image.new('RGB', (input_width, input_high), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = np.random.uniform(0, 1) < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 随机调整颜色
        hue = np.random.uniform(-self.hue, self.hue)
        sat = np.random.uniform(1, self.sat) if np.random.uniform(0, 1) < .5 else 1 / np.random.uniform(1, self.sat)
        val = np.random.uniform(1, self.val) if np.random.uniform(0, 1) < .5 else 1 / np.random.uniform(1, self.val)
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image_data = hsv_to_rgb(x) # numpy array, 0 to 1

        # 矫正缩放后对应的box, box的形状【】
        box_data = np.zeros((self.max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]] * new_width / image_width + dx
            box[:, [1,3]] = box[:, [1,3]] * new_high / image_high + dy
            if flip: box[:, [0,2]] = input_width - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > input_width] = input_width
            box[:, 3][box[:, 3] > input_high] =  input_high
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)] # discard invalid box
            if len(box) > self.max_boxes: box = box[:self.max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data

    def convert_to_tfRecord(self):
        """
        Introduction
        ------------
            把数据集的图片和box以及预处理之后的ground truth box存成tfRecord
        """

        for mode, file in self.file_names.items():
            input_file = os.path.join(self.data_dir, file)
            output_file = os.path.join(self.data_dir, mode + '.tfrecords')
            with tf.python_io.TFRecordWriter(output_file) as record_writer:
                image_data, boxes_data = self.read_annotations(input_file)
                for index in range(len(image_data)):
                    y_true = Preprocess_true_boxes(boxes_data[index], self.input_shape, self.anchors, self.num_classes)
                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data[index].tobytes()])),
                                'box': tf.train.Feature(float_list=tf.train.FloatList(value= boxes_data[index].reshape(-1))),
                                'y_true_13' : tf.train.Feature(float_list = tf.train.FloatList(value = y_true[0].reshape(-1))),
                                'y_true_26': tf.train.Feature(float_list=tf.train.FloatList(value=y_true[0].reshape(-1))),
                                'y_true_52': tf.train.Feature(float_list=tf.train.FloatList(value=y_true[0].reshape(-1)))
                        }))
                    record_writer.write(example.SerializeToString())

    def parser(self, serialized_example):
        """
        Introduction
        ------------
            解析tfRecord
        Parameters
        ----------
            serialized_example: 需要解析的实例
        Returns
        -------

        """
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'box': tf.FixedLenFeature([], tf.float32),
                'y_true_13': tf.FixedLenFeature([], tf.float32),
                'y_true_26': tf.FixedLenFeature([], tf.float32),
                'y_true_52': tf.FixedLenFeature([], tf.float32)
            }
        )
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([416 * 416 * 3])
        image = tf.cast(
            tf.reshape(image, [416, 416, 3]),
            tf.float32)
        box = tf.reshape(features['box'], [20, 5])
        y_true_13 = tf.reshape(features['y_true_13'], [13, 13, 3, 5 + self.num_classes])
        y_true_26 = tf.reshape(features['y_true_26'], [26, 26, 3, 5 + self.num_classes])
        y_true_52 = tf.reshape(features['y_true_52'], [52, 52, 3, 5 + self.num_classes])
        return image, box, y_true_13, y_true_26, y_true_52


    def make_batch(self, mode, batch_size):
        """
        Introduction
        ------------
            读取训练集、验证集、测试集的tfRecord的数据
        Parameters
        ----------
            mode: 标识训练集、验证集和测试集
            batch_size: batch的大小
        Returns
        -------
        """
        Filenames = os.path.join(self.data_dir, mode + '.tfrecords')
        if not os.path.exists(Filenames):
            self.convert_to_tfRecord()
        # 读取TFRecord数据作为数据集
        Dataset = tf.data.TFRecordDataset(filenames = Filenames).repeat()
        # 并行的对数据进行map预处理
        Dataset = Dataset.map(self.parser, num_parallel_calls = config.num_parallel_calls)
        if mode == 'train':
            Dataset = Dataset.shuffle(buffer_size = 16000 + 3 * batch_size)
        Dataset = Dataset.batch(batch_size)
        iterator = Dataset.make_one_shot_iterator()
        image_batch, box_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = iterator.get_next()
        return image_batch, box_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch