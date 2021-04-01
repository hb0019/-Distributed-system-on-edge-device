import tensorflow as tf

weight_init = tf.contrib.layers.xavier_initializer()


def conv2d(input, filter_size, kernel_size=(3, 3), strides=(1, 1), padding='same'):
	return tf.layers.conv2d(input, filter_size, kernel_size, strides=strides, padding=padding,
							kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-4),
							kernel_initializer=weight_init, name='conv2d')


def batch_normalization_layer(inputs, is_training=True, name='bn'):
	tmp = tf.layers.batch_normalization(inputs=inputs, training=is_training, epsilon=1e-3, name=name)
	return tmp


def dense(x, nclass, is_training=True, name=None):
	tmp = tf.layers.dense(x, nclass, name=name, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-4),
						  kernel_initializer=weight_init)
	# tmp = tf.layers.dropout(tmp, rate=0.1, training=is_training)
	return tmp


def global_avg_pooling(x):
	gap = tf.reduce_mean(x, axis=[1, 2])
	return gap


def max_pooling(x, size=(3, 3), strides=(2, 2), padding='same'):
	return tf.layers.max_pooling2d(x, size, strides, padding=padding)


def relu6(x, name='relu6'):
	return tf.nn.relu6(x, name)


def conv2d_bn(x, nb_filter, kernel_size, strides=(1, 1), padding='same', is_training=True):
	"""
    conv2d -> batch normalization -> relu activation
    """
	x = conv2d(x, nb_filter, kernel_size=kernel_size, strides=strides, padding=padding)
	x = batch_normalization_layer(x, is_training=is_training)
	return x


def shortcut(input, residual):
	input_shape = input.get_shape()
	residual_shape = residual.get_shape()
	stride_height = input_shape[1] // residual_shape[1]
	stride_width = input_shape[2] // residual_shape[2]
	equal_channels = input_shape[3] == residual_shape[3]
	identity = input
	if stride_width > 1 or stride_height > 1 or not equal_channels:
		with tf.variable_scope('shutcut'):
			identity = conv2d(input, residual_shape[3], (1, 1), strides=(stride_width, stride_height), padding="valid")

	return identity+residual


def basic_block(nb_filter, strides=(1, 1), is_training=True):
	"""
	基本的ResNet building block，适用于ResNet-18和ResNet-34.
	"""

	def f(input):
		with tf.variable_scope('basic_conv1'):
			conv1 = conv2d_bn(input, nb_filter, kernel_size=(3, 3), strides=strides, is_training=is_training)
			conv1 = tf.nn.relu(conv1)
		with tf.variable_scope('basic_conv2'):
			residual = conv2d_bn(conv1, nb_filter, kernel_size=(3, 3), is_training=is_training)
			output = shortcut(input, residual)
			output = tf.nn.relu(output)
		return output

	return f


def residual_block(nb_filter, repetitions, is_first_layer=False, is_training=True):
	"""
	构建每层的residual模块，对应论文参数统计表中的conv2_x -> conv5_x
	"""

	def f(input):
		for i in range(repetitions):
			strides = (1, 1)
			if i == 0 and not is_first_layer:
				strides = (2, 2)
			with tf.variable_scope('basic_%d' % i):
				input = basic_block(nb_filter, strides, is_training=is_training)(input)
		return input

	return f


def resnet_18(input_, nclass=10, is_training=True, reuse=None):
	"""
	build resnet-18 model using keras with TensorFlow backend.
	:param input_shape: input shape of network, default as (224,224,3)
	:param nclass: numbers of class(output shape of network), default as 1000
	:return: resnet-18 model
	"""
	with tf.variable_scope('network', reuse=reuse):
		with tf.variable_scope('conv1'):
			conv1 = conv2d_bn(input_, 64, kernel_size=(7, 7), strides=(2, 2), is_training=is_training)
			pool1 = max_pooling(conv1, (3, 3), (2, 2), padding='same')
		with tf.variable_scope('conv2'):
			conv2 = residual_block(64, 2, is_first_layer=True, is_training=is_training)(pool1)
		with tf.variable_scope('conv3'):
			conv3 = residual_block(128, 2, is_training=is_training)(conv2)
		with tf.variable_scope('conv4'):
			conv4 = residual_block(256, 2, is_training=is_training)(conv3)
		with tf.variable_scope('conv5'):
			conv5 = residual_block(512, 2, is_training=is_training)(conv4)

		# conv5 = batch_normalization_layer(conv5, is_training=is_training, name='last_bn')
		# conv5 = relu6(conv5)
		pool2 = global_avg_pooling(conv5)
		pool2 = tf.layers.flatten(pool2, name='flatten3')
		output_ = dense(pool2, nclass, is_training=is_training, name='last_dense')

	return output_
