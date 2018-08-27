import os
import time
import config
from dataReader import Reader
import tensorflow as tf
from model.yolo3_model import yolo


def train():
    """
    Introduction
    ------------
        训练模型
    """
    # 指定使用GPU的Index
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index
    train_data = Reader('train', config.data_dir, config.anchors_path, config.num_classes, input_shape = config.input_shape, max_boxes = config.max_boxes, jitter = config.jitter, hue = config.hue, sat = config.sat, cont = config.cont, bri = config.bri)
    val_data = Reader('val', config.data_dir, config.anchors_path, config.num_classes, input_shape = config.input_shape, max_boxes = config.max_boxes)
    images_train, bbox_true_13_train, bbox_true_26_train, bbox_true_52_train = train_data.provide(config.train_batch_size)
    images_val, bbox_true_13_val, bbox_true_26_val, bbox_true_52_val = val_data.provide(config.val_batch_size)

    model = yolo(config.norm_epsilon, config.norm_decay, config.anchors_path, config.classes_path, config.pre_train)
    is_training = tf.placeholder(dtype = tf.bool, shape = [])
    images = tf.placeholder(shape = [None, 416, 416, 3], dtype = tf.float32)
    bbox_true_13 = tf.placeholder(shape = [None, 13, 13, 3, 85], dtype = tf.float32)
    bbox_true_26 = tf.placeholder(shape = [None, 26, 26, 3, 85], dtype = tf.float32)
    bbox_true_52 = tf.placeholder(shape = [None, 52, 52, 3, 85], dtype = tf.float32)
    bbox_true = [bbox_true_13, bbox_true_26, bbox_true_52]
    output = model.yolo_inference(images, config.num_anchors / 3, config.num_classes, is_training)
    loss = model.yolo_loss(output, bbox_true, model.anchors, config.num_classes, config.ignore_thresh)
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, 1000, 0.95, staircase = True)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
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
        if model.pre_train is True:
            load_ops = model.load_weights(tf.global_variables(scope = 'darknet53'), config.darknet53_weights_path)
            sess.run(load_ops)
        summary_writer = tf.summary.FileWriter('./logs', sess.graph)
        tf.train.start_queue_runners(sess = sess)
        for epoch in range(config.Epoch):
            for step in range(int(config.train_num / config.train_batch_size)):
                start_time = time.time()
                images_value, bbox_true_13_value, bbox_true_26_value, bbox_true_52_value = sess.run([images_train, bbox_true_13_train, bbox_true_26_train, bbox_true_52_train])
                train_loss, _ = sess.run([loss, train_op], {images : images_value, bbox_true_13 : bbox_true_13_value, bbox_true_26 : bbox_true_26_value, bbox_true_52 : bbox_true_52_value, is_training : True})
                duration = time.time() - start_time
                examples_per_sec = float(duration) / config.train_batch_size
                format_str = ('Epoch {} step {},  train loss = {} ( {} examples/sec; {} ''sec/batch)')
                print(format_str.format(epoch, step, train_loss, examples_per_sec, duration))
                summary_writer.add_summary(summary = tf.Summary(value = [tf.Summary.Value(tag = "train loss", simple_value = train_loss)]), global_step = step)
                summary_writer.flush()
            for step in range(int(config.val_num / config.val_batch_size)):
                start_time = time.time()
                images_value, bbox_true_13_value, bbox_true_26_value, bbox_true_52_value = sess.run([images_val, bbox_true_13_val, bbox_true_26_val, bbox_true_52_val])
                val_loss = sess.run(loss, {images: images_value, bbox_true_13: bbox_true_13_value, bbox_true_26: bbox_true_26_value, bbox_true_52: bbox_true_52_value , is_training: False})
                duration = time.time() - start_time
                examples_per_sec = float(duration) / config.val_batch_size
                format_str = ('Epoch {} step {}, val loss = {} ({} examples/sec; {} ''sec/batch)')
                print(format_str.format(epoch, step, val_loss, examples_per_sec, duration))
                summary_writer.add_summary(summary = tf.Summary(value = [tf.Summary.Value(tag = "val loss", simple_value = val_loss)]), global_step = step)
                summary_writer.flush()
            # 每3个epoch保存一次模型
            if epoch % 3 == 0:
                checkpoint_path = os.path.join(config.model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = epoch)



def dstributed_train(ps_hosts, worker_hosts, job_name, task_index):
    """
    Introduction
    ------------
        分布式训练
    Parameters
    ----------
        ps_hosts: sever的host
        worker_hosts: worker的host
        job_name: 判断是作为ps还是worker
        task_index: 任务index
    """
    ps_hosts = ps_hosts.split(',')
    worker_hosts = worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name = job_name, task_index = task_index)
    if job_name == 'ps':
        server.join()
    else:
        with tf.device(tf.train.replica_device_setter(worker_device = "/job:worker/task:%d" % task_index, cluster = cluster)):
            train_data = Reader('train', config.data_dir, config.anchors_path, config.num_classes,
                                input_shape=config.input_shape, max_boxes=config.max_boxes, jitter=config.jitter,
                                hue=config.hue, sat=config.sat, cont=config.cont, bri=config.bri)
            val_data = Reader('val', config.data_dir, config.anchors_path, config.num_classes,
                              input_shape=config.input_shape, max_boxes=config.max_boxes)
            images_train, bbox_true_13_train, bbox_true_26_train, bbox_true_52_train = train_data.provide(config.train_batch_size)
            images_val, bbox_true_13_val, bbox_true_26_val, bbox_true_52_val = val_data.provide(config.val_batch_size)

            model = yolo(config.norm_epsilon, config.norm_decay, config.anchors_path, config.classes_path, config.pre_train)
            is_training = tf.placeholder(dtype=tf.bool, shape=[])
            images = tf.placeholder(shape=[None, 416, 416, 3], dtype=tf.float32)
            bbox_true_13 = tf.placeholder(shape=[None, 13, 13, 3, 85], dtype=tf.float32)
            bbox_true_26 = tf.placeholder(shape=[None, 26, 26, 3, 85], dtype=tf.float32)
            bbox_true_52 = tf.placeholder(shape=[None, 52, 52, 3, 85], dtype=tf.float32)
            bbox_true = [bbox_true_13, bbox_true_26, bbox_true_52]
            output = model.yolo_inference(images, config.num_anchors / 3, config.num_classes, is_training)
            loss = model.yolo_loss(output, bbox_true, model.anchors, config.num_classes, config.ignore_thresh)
            tf.summary.scalar('loss', loss)
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, 1000, 0.95, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # 如果读取预训练权重，则冻结darknet53网络的变量
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if config.pre_train:
                    train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo')
                    train_op = optimizer.minimize(loss=loss, global_step=global_step, var_list=train_var)
                else:
                    train_op = optimizer.minimize(loss=loss, global_step=global_step)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                ckpt = tf.train.get_checkpoint_state(config.model_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    print('restore model', ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    sess.run(init)
                if model.pre_train is True:
                    load_ops = model.load_weights(tf.global_variables(scope='darknet53'), config.darknet53_weights_path)
                    sess.run(load_ops)
                summary_writer = tf.summary.FileWriter('./logs', sess.graph)
                tf.train.start_queue_runners(sess=sess)
                for epoch in range(config.Epoch):
                    for step in range(int(config.train_num / config.train_batch_size)):
                        start_time = time.time()
                        images_value, bbox_true_13_value, bbox_true_26_value, bbox_true_52_value = sess.run(
                            [images_train, bbox_true_13_train, bbox_true_26_train, bbox_true_52_train])
                        train_loss, _ = sess.run([loss, train_op],
                                                 {images: images_value, bbox_true_13: bbox_true_13_value,
                                                  bbox_true_26: bbox_true_26_value, bbox_true_52: bbox_true_52_value,
                                                  is_training: True})
                        duration = time.time() - start_time
                        examples_per_sec = float(duration) / config.train_batch_size
                        format_str = ('Epoch {} step {},  train loss = {} ( {} examples/sec; {} ''sec/batch)')
                        print(format_str.format(epoch, step, train_loss, examples_per_sec, duration))
                        summary_writer.add_summary(
                            summary=tf.Summary(value=[tf.Summary.Value(tag="train loss", simple_value=train_loss)]),
                            global_step=step)
                        summary_writer.flush()
                    for step in range(int(config.val_num / config.val_batch_size)):
                        start_time = time.time()
                        images_value, bbox_true_13_value, bbox_true_26_value, bbox_true_52_value = sess.run(
                            [images_val, bbox_true_13_val, bbox_true_26_val, bbox_true_52_val])
                        val_loss = sess.run(loss, {images: images_value, bbox_true_13: bbox_true_13_value,
                                                   bbox_true_26: bbox_true_26_value, bbox_true_52: bbox_true_52_value,
                                                   is_training: False})
                        duration = time.time() - start_time
                        examples_per_sec = float(duration) / config.val_batch_size
                        format_str = ('Epoch {} step {}, val loss = {} ({} examples/sec; {} ''sec/batch)')
                        print(format_str.format(epoch, step, val_loss, examples_per_sec, duration))
                        summary_writer.add_summary(
                            summary=tf.Summary(value=[tf.Summary.Value(tag="val loss", simple_value=val_loss)]),
                            global_step=step)
                        summary_writer.flush()
                    # 每3个epoch保存一次模型
                    if epoch % 3 == 0:
                        checkpoint_path = os.path.join(config.model_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=epoch)

if __name__ == "__main__":
    train()








