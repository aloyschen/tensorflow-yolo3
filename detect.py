import config
import argparse
import numpy as np
import tensorflow as tf
from yolo_predict import yolo_predictor
from PIL import Image, ImageFont, ImageDraw
from utils import letterbox_image, load_weights


def detect(image_path, model_path, yolo_weights = None):
    """
    Introduction
    ------------
        加载模型，进行预测
    Parameters
    ----------
        model_path: 模型路径
        image_path: 图片路径
    """
    image = Image.open(image_path)
    resize_image = letterbox_image(image, (416, 416))
    image_data = np.array(resize_image, dtype = np.float32)
    image_data /= 255.
    image_data = np.expand_dims(image_data, axis = 0)
    input_image_shape = tf.placeholder(dtype = tf.int32, shape = (2,))
    input_image = tf.placeholder(shape = [None, 416, 416, 3], dtype = tf.float32)
    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)
    boxes, scores, classes = predictor.predict(input_image, input_image_shape)
    with tf.Session() as sess:
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            load_op = load_weights(tf.global_variables(scope = 'predict'), weights_file = yolo_weights)
            sess.run(load_op)
        else:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                input_image: image_data,
                input_image_shape: [image.size[1], image.size[0]]
            })
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        font = ImageFont.truetype(font = 'font/FiraMono-Medium.otf', size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = predictor.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline = predictor.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill = predictor.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        image.show()
        image.save('./result1.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default = argparse.SUPPRESS)
    parser.add_argument(
        '--image_file', type = str, help = 'image file path'
    )
    FLAGS = parser.parse_args()
    if config.pre_train_yolo3 == True:
        detect(FLAGS.image_file, config.model_dir, config.yolo3_weights_path)
    else:
        detect(FLAGS.image_file, config.model_dir)
