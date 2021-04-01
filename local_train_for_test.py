import tensorflow as tf
import os
import matplotlib.pyplot as plt
import math
import time
import json
import tensorflow.contrib.slim as slim
import cifar_10_data
import CLS_64_data
import resnet_test
import resnet
import vgg
import googlenet
import cnn_model
from tensorflow.python.framework import graph_util
import numpy as np

learning_rate_tactics = {"exponential_decay": False, "piecewise_constant": False, "natural_exp_decay": False,
                         "polynomial_decay": False, "inverse_time_decay": False, "cosine_decay": True}
flags = tf.app.flags
IMAGE_PIXELS = 32
classes_num = 10
CHANNEL = 3
# 定义默认训练参数和数据路径
flags.DEFINE_string('data_dir', 'data', 'Directory  for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('batch_size', 128, 'Training batch size ')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_integer('training_epochs', 100, 'total training epochs')
FLAGS = flags.FLAGS

Bert_Use_GPU = True
if Bert_Use_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用GPU0
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 使用CPU


def main(unused_argv):
    def Evaluate(sess, loss, accuracy):
        test_acc = 0.0
        test_loss = 0.0
        test_pre_index = 0
        # add = 500
        # test_iteration = 20
        add = FLAGS.batch_size
        test_iteration = int(len(test_data) / add)

        for _ in range(test_iteration):
            test_batch_x = test_data[test_pre_index: test_pre_index + add]
            test_batch_y = test_labels[test_pre_index: test_pre_index + add]
            test_pre_index = test_pre_index + add

            test_feed_dict = {
                input_images: test_batch_x,
                input_labels: test_batch_y,
                keep_prob: 1.0
                # is_training: False
            }
            loss_, acc_ = sess.run([loss, accuracy], feed_dict=test_feed_dict)

            test_loss += loss_
            test_acc += acc_

        return test_acc / test_iteration, test_loss / test_iteration

    trains, labels, test_data, test_labels = cifar_10_data.prepare_data()
    trains, test_data = cifar_10_data.color_preprocessing(trains, test_data)
    # trains, labels, test_data, test_labels = CLS_64_data.prepare_data()

    train_size, test_size = len(labels), len(test_labels)
    total_batch = train_size // FLAGS.batch_size + 1
    print("train size: %d  test size: %d  total batch: %d" % (train_size, test_size, total_batch))

    input_images = tf.placeholder(tf.float32, [None, IMAGE_PIXELS, IMAGE_PIXELS, CHANNEL], name='input')
    input_labels = tf.placeholder(tf.float32, [None, classes_num], name='label')
    # input_images = tf.image.per_image_standardization(input_images) # 图片标准化处理

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # is_training = tf.placeholder(tf.bool, name='is_training')
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    # graph start
    # trainingy
    # y, _ = cnn_model.mobilenet_v3_small(input_images, classes_num)
    # y = resnet.resnet_18(input_images, classes_num)
    y = resnet_test.inference(input_images, classes_num, training=True, filters=16, n=5, ver=2)
    # y = vgg.vgg16(image=input_images, classes_num=classes_num, keep_prob=keep_prob, is_training=is_training)
    # graph end
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=input_labels, logits=y, pos_weight=2.0))
    weight_decay = 5e-4
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
    loss = cost + l2_loss
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(input_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.01)
    opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train = opt.minimize(loss, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables())
    log_path = r'C:\Users\ASUS\Desktop\distributed\model'
    # sv = tf.train.Supervisor(logdir=log_path, init_op=init_op, saver=saver)
    collections = {'time': [], 'train_acc': [], 'loss': [], 'test_acc': [], 'test_loss': [], 'learning_rate': []}

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        # 定义总参数量、可训练参数量及非可训练参数量变量
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        # 遍历tf.global_variables()返回的全局变量列表
        for var in tf.global_variables():
            shape = var.shape
            array = np.asarray([dim.value for dim in shape])
            mulValue = np.prod(array)
            Total_params += mulValue  # 总参数量
            if var.trainable:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量
        print(f'Total params: {4 * Total_params / 1024 / 1024}')
        print(f'Trainable params: {4 * Trainable_params / 1024 / 1024}')
        print(f'Non-trainable params: {4 * NonTrainable_params / 1024 / 1024}')

        para_num = sum([np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
        # para_size: 参数个数 * 每个4字节(float32) / 1024 / 1024，单位为 MB
        para_size = para_num * 4 / 1024 / 1024
        print("model size: ", para_size)

        time_begin = time.time()
        print('Traing begins @ %f' % time_begin)
        for epoch in range(1, FLAGS.training_epochs + 1):
            batch_time = 0
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            # 准备数据

            for step in range(total_batch):
                if pre_index <= train_size:
                    batch_x = trains[pre_index: pre_index + FLAGS.batch_size]
                    batch_y = labels[pre_index: pre_index + FLAGS.batch_size]
                else:
                    batch_x = trains[pre_index:]
                    batch_y = labels[pre_index:]
                pre_index += FLAGS.batch_size

                batch_x = cifar_10_data.data_augmentation(batch_x)

                # global_step_ = min(((epoch-1)*total_batch+step+1), 80*total_batch)
                # cosine_decay_ = 0.5*(1+math.cos(math.pi*global_step_/(80*total_batch)))
                # decayed_ = (1-5*1e-4)*cosine_decay_ + 5*1e-4
                # learning_rate_ = 1e-3*decayed_

                # learning_rate_ = 1e-3 * 0.5 ** (((epoch - 1) * total_batch + step + 1) / (120 * total_batch))

                if ((epoch - 1) * total_batch + step + 1)<(20*total_batch):
                    learning_rate_ = 0.1
                elif ((epoch - 1) * total_batch + step + 1)<(60*total_batch):
                    learning_rate_ = 0.01
                else:
                    learning_rate_ = 0.001

                train_feed_dict = {
                    input_images: batch_x,
                    input_labels: batch_y,
                    learning_rate: learning_rate_,
                    keep_prob: 1.0
                }
                _, batch_loss, batch_acc = sess.run([train, loss, accuracy], feed_dict=train_feed_dict)

                train_loss += batch_loss
                train_acc += batch_acc
                if not step % 100:
                    batch_time, cost_time = time.time() - time_begin, time.time() - time_begin - batch_time
                    information = 'batch accuracy: %f  cost time: %f  step: %d  learning rate: %f ' % (
                        batch_acc, cost_time, step, learning_rate_)
                    print(information)

            train_loss /= total_batch  # average loss
            train_acc /= total_batch  # average accuracy
            now_time = time.time()
            test_acc_, test_loss_ = 0, 0
            test_acc_, test_loss_ = Evaluate(sess, loss, accuracy)

            line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f, test_loss: %.4f, use_time: %.4f, learning_rate: %f \n" % (
                epoch, FLAGS.training_epochs, train_loss, train_acc, test_acc_, test_loss_, now_time - time_begin,
                learning_rate_)
            print(line)
            collections['time'].append(now_time - time_begin)
            collections['train_acc'].append(train_acc)
            collections['loss'].append(train_loss)
            collections['test_acc'].append(float(test_acc_))
            collections['test_loss'].append(float(test_loss_))
            collections['learning_rate'].append(float(learning_rate_))

        saver.save(sess=sess, save_path='log' + '/model.ckpt', global_step=global_step)

        with open('test_data.json', 'wb') as file:
            file.write(json.dumps(collections).encode())

    # draw this train result
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(collections['time'], collections['test_acc'], color='darkblue', linestyle="-", marker=".", linewidth=1)
    ax1.plot(collections['time'], collections['train_acc'], color='darkred', linestyle="-", marker=".", linewidth=1)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("train result")
    ax1.set_xlabel('Time(sec)')
    ax1.legend(['test_acc', 'train_acc'], loc='center right')

    ax2 = ax1.twinx()
    ax2.plot(collections['time'], collections['learning_rate'], color='darkgray', linestyle="-", marker=".",
             linewidth=1)
    ax2.set_ylabel("Learning rate")
    ax2.set_xlabel('Time(sec)')
    ax2.legend(['learning_rate'], loc='lower center')

    # plt.savefig(path + "/" + name + '-' + str(file_index) + '.pdf', dpi=600)
    plt.show()

    # draw this loss&learning_rate
    fig = plt.figure()
    # slope of Loss&Learning
    '''
    log_lr = list(map(math.log,collections['learning_rate']))
    log_loss = list(map(math.log,collections['loss']))
    useful_log_loss = [log_loss[0]]+[log_loss[i] for i in range(1,len(log_loss)) if log_loss[i]<=log_loss[i-1]]
    plt.plot([i for i in range(1,len(useful_log_loss))],[abs((useful_log_loss[i]-useful_log_loss[i-1])/(log_lr[i]-log_lr[i-1]),marker=".", linewidth=1)
    plt.ylabel('slope of Loss&Learning')
    plt.xlabel('iteration')
    plt.show()
    '''
    # slope_lr = []
    plt.subplot(211)
    plt.loglog(collections['learning_rate'], collections['loss'], color='darkblue', linestyle="-", marker=".",
               linewidth=1)
    plt.xlabel('Learning rate(log)')
    plt.ylabel('Loss(log)')
    # slope of Loss&iternation
    # slop_le = []
    plt.subplot(212)
    plt.loglog([element for element in range(len(collections['loss']))], collections['loss'], color='darkblue',
               linestyle="-", marker=".", linewidth=1)
    plt.semilogy()  # 半对数坐标
    plt.xlabel('Train epoch')
    plt.ylabel('Loss(log)')
    plt.show()


if __name__ == '__main__':
    tf.app.run()
