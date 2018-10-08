import os
import time
import config
import numpy as np
from PIL import Image
import tensorflow as tf
from dataReader import Reader
from model.yolo3_model import yolo
from collections import defaultdict
from yolo_predict import yolo_predictor
from utils import draw_box, load_weights, letterbox_image, voc_ap

# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index

def train():
    """
    Introduction
    ------------
        训练模型
    """
    train_reader = Reader('train', config.data_dir, config.anchors_path, config.num_classes, input_shape = config.input_shape, max_boxes = config.max_boxes)
    train_data = train_reader.build_dataset(config.train_batch_size)
    is_training = tf.placeholder(tf.bool, shape = [])
    iterator = train_data.make_one_shot_iterator()
    images, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = iterator.get_next()
    images.set_shape([None, config.input_shape, config.input_shape, 3])
    bbox.set_shape([None, config.max_boxes, 5])
    grid_shapes = [config.input_shape // 32, config.input_shape // 16, config.input_shape // 8]
    bbox_true_13.set_shape([None, grid_shapes[0], grid_shapes[0], 3, 5 + config.num_classes])
    bbox_true_26.set_shape([None, grid_shapes[1], grid_shapes[1], 3, 5 + config.num_classes])
    bbox_true_52.set_shape([None, grid_shapes[2], grid_shapes[2], 3, 5 + config.num_classes])
    draw_box(images, bbox)
    model = yolo(config.norm_epsilon, config.norm_decay, config.anchors_path, config.classes_path, config.pre_train)
    bbox_true = [bbox_true_13, bbox_true_26, bbox_true_52]
    output = model.yolo_inference(images, config.num_anchors / 3, config.num_classes, is_training)
    loss = model.yolo_loss(output, bbox_true, model.anchors, config.num_classes, config.ignore_thresh)
    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss
    tf.summary.scalar('loss', loss)
    merged_summary = tf.summary.merge_all()
    global_step = tf.Variable(0, trainable = False)
    lr = tf.train.exponential_decay(config.learning_rate, global_step, decay_steps = 2000, decay_rate = 0.8)
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    # 如果读取预训练权重，则冻结darknet53网络的变量
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if config.pre_train:
            train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo')
            train_op = optimizer.minimize(loss = loss, global_step = global_step, var_list = train_var)
        else:
            train_op = optimizer.minimize(loss = loss, global_step = global_step)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session(config = tf.ConfigProto(log_device_placement = False)) as sess:
        ckpt = tf.train.get_checkpoint_state(config.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('restore model', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)
        if config.pre_train is True:
            load_ops = load_weights(tf.global_variables(scope = 'darknet53'), config.darknet53_weights_path)
            sess.run(load_ops)
        summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)
        loss_value = 0
        for epoch in range(config.Epoch):
            for step in range(int(config.train_num / config.train_batch_size)):
                start_time = time.time()
                train_loss, summary, global_step_value, _ = sess.run([loss, merged_summary, global_step, train_op], {is_training : True})
                loss_value += train_loss
                duration = time.time() - start_time
                examples_per_sec = float(duration) / config.train_batch_size
                format_str = ('Epoch {} step {},  train loss = {} ( {} examples/sec; {} ''sec/batch)')
                print(format_str.format(epoch, step, loss_value / global_step_value, examples_per_sec, duration))
                summary_writer.add_summary(summary = tf.Summary(value = [tf.Summary.Value(tag = "train loss", simple_value = train_loss)]), global_step = step)
                summary_writer.add_summary(summary, step)
                summary_writer.flush()
            # 每3个epoch保存一次模型
            if epoch % 3 == 0:
                checkpoint_path = os.path.join(config.model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = global_step)


def eval(model_path, min_Iou = 0.5, yolo_weights = None):
    """
    Introduction
    ------------
        计算模型在coco验证集上的MAP, 用于评价模型
    """
    ground_truth = {}
    class_pred = defaultdict(list)
    gt_counter_per_class = defaultdict(int)
    input_image_shape = tf.placeholder(dtype = tf.int32, shape = (2,))
    input_image = tf.placeholder(shape = [None, 416, 416, 3], dtype = tf.float32)
    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)
    boxes, scores, classes = predictor.predict(input_image, input_image_shape)
    val_Reader = Reader("val", config.data_dir, config.anchors_path, config.num_classes, input_shape = config.input_shape, max_boxes = config.max_boxes)
    image_files, bboxes_data = val_Reader.read_annotations()
    with tf.Session() as sess:
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            load_op = load_weights(tf.global_variables(scope = 'predict'), weights_file = yolo_weights)
            sess.run(load_op)
        else:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
        for index in range(len(image_files)):
            val_bboxes = []
            image_file = image_files[index]
            file_id = os.path.split(image_file)[-1].split('.')[0]
            for bbox in bboxes_data[index]:
                left, top, right, bottom, class_id = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
                class_name = val_Reader.class_names[int(class_id)]
                bbox = [float(left), float(top), float(right), float(bottom)]
                val_bboxes.append({"class_name" : class_name, "bbox": bbox, "used": False})
                gt_counter_per_class[class_name] += 1
            ground_truth[file_id] = val_bboxes
            image = Image.open(image_file)
            resize_image = letterbox_image(image, (416, 416))
            image_data = np.array(resize_image, dtype = np.float32)
            image_data /= 255.
            image_data = np.expand_dims(image_data, axis = 0)

            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict = {
                    input_image : image_data,
                    input_image_shape : [image.size[1], image.size[0]]
                })
            print("detect {}/{} found boxes: {}".format(index, len(image_files), len(out_boxes)))
            for o, c in enumerate(out_classes):
                predicted_class = val_Reader.class_names[c]
                box = out_boxes[o]
                score = out_scores[o]

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                bbox = [left, top, right, bottom]
                class_pred[predicted_class].append({"confidence": str(score), "file_id": file_id, "bbox": bbox})

    # 计算每个类别的AP
    sum_AP = 0.0
    count_true_positives = {}
    for class_index, class_name in enumerate(sorted(gt_counter_per_class.keys())):
        count_true_positives[class_name] = 0
        predictions_data = class_pred[class_name]
        # 该类别总共有多少个box
        nd = len(predictions_data)
        tp = [0] * nd  # true positive
        fp = [0] * nd  # false positive
        for idx, prediction in enumerate(predictions_data):
            file_id = prediction['file_id']
            ground_truth_data = ground_truth[file_id]
            bbox_pred = prediction['bbox']
            Iou_max = -1
            gt_match = None
            for obj in ground_truth_data:
                if obj['class_name'] == class_name:
                    bbox_gt = obj['bbox']
                    bbox_intersect = [max(bbox_pred[0], bbox_gt[0]), max(bbox_gt[1], bbox_pred[1]), min(bbox_gt[2], bbox_pred[2]), min(bbox_gt[3], bbox_pred[3])]
                    intersect_weight = bbox_intersect[2] - bbox_intersect[0] + 1
                    intersect_high = bbox_intersect[3] - bbox_intersect[1] + 1
                    if intersect_high > 0 and intersect_weight > 0:
                        union_area = (bbox_pred[2] - bbox_pred[0] + 1) * (bbox_pred[3] - bbox_pred[1] + 1) + (bbox_gt[2] - bbox_gt[0] + 1) * (bbox_gt[3] - bbox_gt[1] + 1) - intersect_weight * intersect_high
                        Iou = intersect_high * intersect_weight / union_area
                        if Iou > Iou_max:
                            Iou_max = Iou
                            gt_match = obj
            if Iou_max > min_Iou:
                if not gt_match['used'] and gt_match is not None:
                    tp[idx] = 1
                    gt_match['used'] = True
                else:
                    fp[idx] = 1
            else:
                fp[idx] = 1
        # 计算精度和召回率
        sum_class = 0
        for idx, val in enumerate(fp):
            fp[idx] += sum_class
            sum_class += val
        sum_class = 0
        for idx, val in enumerate(tp):
            tp[idx] += sum_class
            sum_class += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = tp[idx] / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = tp[idx] / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec, prec)
        sum_AP += ap
    MAP = sum_AP / len(gt_counter_per_class) * 100
    print("The Model Eval MAP: {}".format(MAP))

if __name__ == "__main__":
    train()
    # 计算模型的Map
    # eval(config.model_dir, yolo_weights = config.yolo3_weights_path)

