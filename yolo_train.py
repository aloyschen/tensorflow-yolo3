import os
import config
from dataReader import Reader
import tensorflow as tf
from model.yolo3_model import yolo


def input_fn(data_dir, subset, batch_size):
    """
    Introduction
    ------------
        构建模型的输入模块
    Parameters
    ----------
        data_dir: 训练集数据路径
        subset: 训练集，验证集
        batch_size: batch大小
    Returns
    -------

    """
    with tf.device('/cpu:0'):
        dataReader = Reader(data_dir, config.anchors_path, config.num_classes, input_shape = config.input_shape, max_boxes = config.max_boxes, jitter = config.jitter, hue = config.hue, sat = config.sat, cont = config.cont, bri = config.bri)
        image_batch, bbox_true_13, bbox_true_26, bbox_true_52 = dataReader.make_batch(subset, batch_size)
        return image_batch, [bbox_true_13, bbox_true_26, bbox_true_52]


def model_fn(features, labels, mode):
    """
    Introduction
    ------------
        构建模型的Estimator
    Parameters
    ----------
        features: 数据特征
        mode: 模型训练或者验证
    Returns
    -------
        Estimator: 自定义模型Estimator
    """
    model = yolo(config.norm_epsilon, config.norm_decay, config.anchors_path, config.classes_path, config.pre_train, config.weights_file)
    output = model.yolo_inference(features, config.num_anchors / 3, config.num_classes, config.training)
    loss = model.yolo_loss(output, labels, model.anchors, config.num_classes, config.ignore_thresh)
    tf.summary.scalar('loss', loss)
    boundaries = [1000, 1500, 2500]
    staged_lr = [config.learning_rate * x for x in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(), boundaries, staged_lr)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
    tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss,train_op=train_op, training_hooks = [logging_hook])


def train():
    """
    Introduction
    ------------
        训练函数
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index
    tf.logging.set_verbosity(tf.logging.INFO)
    model = tf.estimator.Estimator(model_fn = model_fn, model_dir = config.model_dir)
    train_input_spec = tf.estimator.TrainSpec(input_fn = lambda: input_fn(config.data_dir, 'train', config.train_batch_size), max_steps = config.train_num // config.train_batch_size * config.Epoch)
    val_input_spec = tf.estimator.EvalSpec(input_fn = lambda: input_fn(config.data_dir, 'val', config.val_batch_size),
        steps = config.val_num // config.val_batch_size * config.Epoch)
    tf.estimator.train_and_evaluate(model, train_input_spec, val_input_spec)

if __name__ == "__main__":
    train()








