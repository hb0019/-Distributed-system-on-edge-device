import tensorflow as tf



def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME', name=name)


def batch_norm(input, is_training):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=is_training, updates_collections=None)


def vgg16(image, classes_num, keep_prob=0.5, is_training=True):
    #
    W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_1 = bias_variable([64])
    output = tf.nn.relu(batch_norm(conv2d(image, W_conv1_1) + b_conv1_1, is_training))


    W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_2 = bias_variable([64])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv1_2) + b_conv1_2, is_training))
    output = max_pool(output, 2, 2, "pool1")


    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_1 = bias_variable([128])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_1) + b_conv2_1, is_training))


    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_2 = bias_variable([128])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_2) + b_conv2_2, is_training))
    output = max_pool(output, 2, 2, "pool2")


    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_1 = bias_variable([256])
    output = tf.nn.relu( batch_norm(conv2d(output,W_conv3_1) + b_conv3_1, is_training))


    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_2 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_2) + b_conv3_2, is_training))


    W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_3 = bias_variable([256])
    output = tf.nn.relu( batch_norm(conv2d(output, W_conv3_3) + b_conv3_3, is_training))
    output = max_pool(output, 2, 2, "pool3")


    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_1 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_1) + b_conv4_1, is_training))


    W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_2 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_2) + b_conv4_2, is_training))


    W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_3 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_3) + b_conv4_3, is_training))
    output = max_pool(output, 2, 2)


    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_1 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_1) + b_conv5_1, is_training))


    W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_2 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_2) + b_conv5_2, is_training))


    W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_3 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_3) + b_conv5_3, is_training))
    #output = max_pool(output, 2, 2)


    # output = tf.contrib.layers.flatten(output)
    output = tf.reshape(output, [-1, 2*2*512])


    W_fc1 = tf.get_variable('fc1', shape=[2048, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc1 = bias_variable([4096])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc1) + b_fc1, is_training))
    output = tf.nn.dropout(output, keep_prob)


    W_fc2 = tf.get_variable('fc7', shape=[4096, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc2 = bias_variable([4096])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc2) + b_fc2, is_training))
    output = tf.nn.dropout(output, keep_prob)


    W_fc3 = tf.get_variable('fc3', shape=[4096, classes_num], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc3 = bias_variable([10])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc3) + b_fc3, is_training))

    return output