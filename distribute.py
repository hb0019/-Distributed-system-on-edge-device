import time
import tensorflow as tf
import json
import math
import os
import cnn_model
import resnet_test
import resnet
import cifar_10_data
import CLS_64_data
import experimental as em

class Distribute:
	IMAGE_PIXELS = 32
	classes_num = 10
	CHANNEL = 3

	def __init__(self, **kwargs):
		"""
		example:
		kwargs = {
		'job_name':'ps',
		'task_index':0,
		'batch_size':2048,
		'ps_hosts':['192.168.0.104:22221'],
		'worker_hosts':['192.168.0.100:22221','192.168.0.101:22221'],
		'training_epochs':5,
		'learning_rate':1e3,
		'train_steps':1200
		}
		:param kwargs:
		"""
		for k, v in kwargs.items():
			setattr(self, k, v)
		print(self.__class__.__name__, 'is starting!')
		# select whether use GPU
		self.Bert_Use_GPU = False
		if self.Bert_Use_GPU:
			os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用GPU0
		else:
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 使用CPU

	def main(self):
		if self.job_name == "worker":
			trains, labels, validation_data, validation_labels = cifar_10_data.prepare_data()
			# trains, labels, validation_data, validation_labels = CLS_64_data.prepare_data()
			train_size = len(labels)
			total_batch = int(train_size / self.batch_size) + 1
			print('train_size : %d' % train_size, '    total_batch : %d' % total_batch)
			trains, validation_data = cifar_10_data.color_preprocessing(trains, validation_data)

		if self.job_name is None or self.job_name == '':
			raise ValueError('Must specify an explicit job_name !')
		else:
			print('job_name : %s' % self.job_name)
		if self.task_index is None or self.task_index == '':
			raise ValueError('Must specify an explicit task_index!')
		else:
			print('task_index : %d' % self.task_index)

		ps_spec = self.ps_hosts  # list of char
		worker_spec = self.worker_hosts  # list of char

		# 创建集群
		num_worker = len(worker_spec)
		cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})  # define the whole network configuration
		server = tf.train.Server(cluster, job_name=self.job_name, task_index=self.task_index)

		is_chief = (self.task_index == 0)

		if self.job_name == 'ps':
			server.join()
		elif self.job_name == "worker":
			batch_information_file = "batch_information.txt"
			record_handle = em.Experiment_record()
			def Evaluate():
				test_acc = 0.0
				test_pre_index = 0
				add = self.batch_size
				test_iteration = int(len(validation_data) / add)

				for _ in range(test_iteration):
					test_batch_x = validation_data[test_pre_index: test_pre_index + add]
					test_batch_y = validation_labels[test_pre_index: test_pre_index + add]
					test_pre_index = test_pre_index + add

					test_feed_dict = {
						x: test_batch_x,
						y_: test_batch_y
					}

					acc_ = sess.run(accuracy, feed_dict=test_feed_dict)

					test_acc += acc_
				return test_acc / test_iteration 

			if self.Bert_Use_GPU:
				worker_device = "/job:worker/task:%d" % self.task_index
			else:
				worker_device = "/job:worker/task:%d/cpu:0" % self.task_index
			with tf.device(tf.train.replica_device_setter(
                            worker_device=worker_device, cluster=cluster)):

				x = tf.placeholder(tf.float32, [None, self.IMAGE_PIXELS, self.IMAGE_PIXELS, self.CHANNEL], name='input')
				y_ = tf.placeholder(tf.float32, [None, self.classes_num], name='label')

				learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

				# graph start
				# training
				# y = cnn_model.CNN(x)
				# y = resnet.resnet_18(x, self.classes_num)
				y = resnet_test.inference(x, 10, training=True, filters=16, n=5, ver=2)
				# graph end
				cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
				weight_decay = 5e-4
				l2_loss = weight_decay * tf.add_n(
					# loss is computed using fp32 for numerical stability.
					[tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
				loss = cost + l2_loss

				# 精确度
				correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

				global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量

				# 固定学习率
				# opt = tf.train.AdamOptimizer(self.learning_rate)
				# 余弦衰减学习率
				# lr, DecaySteps = 1e-1, 80*total_batch
				# learning_rate = tf.train.cosine_decay(
				# learning_rate=lr, global_step=global_step, decay_steps=DecaySteps, alpha= 1e-3)
				opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

				with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
					train_step = opt.minimize(loss, global_step=global_step)

				# 生成本地的参数初始化操作init_op
				init_op = tf.global_variables_initializer()
				train_dir = 'log'
				saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
				sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, saver=saver, init_op=init_op,
										 recovery_wait_secs=1,
										 save_model_secs=60,
										 global_step=global_step)

				if is_chief:
					print('Worker %d: Initailizing session...' % self.task_index)
				else:
					print('Worker %d: Waiting for session to be initaialized...' % self.task_index)

				with sv.managed_session(master=server.target) as sess:
					print('Worker %d: Session initialization  complete.' % self.task_index)

					time_begin = time.time()
					print('Traing begins @ %f' % time_begin)
					collections = {'time': [], 'train_acc': [], 'test_acc': []}
					local_step, late_batch_step = 0, 0
					train_acc = list()
					step = sess.run(global_step)
					while (step+1)//total_batch+1 < self.training_epochs:
						batch_count = (step+1)%total_batch

						# Compute the offset of the current minibatch in the data.
						offset = (batch_count * self.batch_size) % (train_size)
						batch_xs = trains[offset:offset + self.batch_size]
						batch_ys = labels[offset:offset + self.batch_size]
						batch_xs = cifar_10_data.data_augmentation(batch_xs)

						# 余弦衰减
						# global_step_ = min(step+1, 80*total_batch)
						# cosine_decay_ = 0.5*(1+math.cos(math.pi*global_step_/(80*total_batch)))
						# decayed_ = (1-1e-3)*cosine_decay_ + 1e-3
						# learning_rate_ = 1e-1*decayed_

						# 阶梯衰减
						if step+1 < (20*total_batch):
							learning_rate_ = 0.1
						elif step+1 < (30*total_batch):
							learning_rate_ = 0.02
						elif step+1 < (45*total_batch):
							learning_rate_ = 0.01
						else:
							learning_rate_ = 0.001

						train_feed = {
							x: batch_xs,
							y_: batch_ys,
							learning_rate: learning_rate_
							}
						if sv.should_stop():
							break

						_, step, batch_acc = sess.run([train_step, global_step, accuracy], feed_dict=train_feed)

						train_acc.append(batch_acc)
						now = time.time()
						local_step += 1

						line = '%f: Worker %d: traing step %d done (global step:%d) epoch: %d--batch: %d '% (
							now, self.task_index, local_step, step, (step+1)//total_batch+1, batch_count)
						print(line)
						package = 'time: %f; local_step: %d \n' %(now, local_step)
						record_handle.record_data(batch_information_file, package)

						if not is_chief and not step%100:
							saver.save(sess=sess, save_path=train_dir + '/model.ckpt', global_step=global_step)

						if batch_count == 0:
							train_acc_value = sum(train_acc)/len(train_acc)
							collections['train_acc'].append(train_acc_value)
							print('train accuracy is:%f' % train_acc_value)

							train_time = time.time() - time_begin
							print('Training elapsed time of this epoch: %f s' % train_time)
							collections['time'].append(train_time)
							# validation
							test_acc = Evaluate()
							line = "After %d training step(s), train_acc: %.4f, test_acc: %.4f\n" % (
								step, train_acc_value, test_acc)
							print(line)
							collections['test_acc'].append(test_acc)
							train_acc= []

							path = record_handle.get_record_path()
							with open(path + '/' + 'distributed_section_data.json', 'wb') as file:
								file.write(json.dumps(collections).encode())
						if step%total_batch < late_batch_step%total_batch:
							train_acc = []
						late_batch_step = step
						step = sess.run(global_step)

					try:
						train_acc_value = sum(train_acc)/len(train_acc)
						test_acc = Evaluate()
					except:
						train_acc_value = test_acc = Evaluate()
					collections['train_acc'].append(train_acc_value)
					collections['time'].append(time.time() - time_begin)
					collections['test_acc'].append(test_acc)
					path = record_handle.get_record_path()
					with open(path + '/' + 'distributed_section_data.json', 'wb') as file:
						file.write(json.dumps(collections).encode())

					cmd = "python3 experimental.py %s %d" % ("kill", os.getpid())
					print("Training finish!")
					os.system(cmd)



if __name__ == '__main__':
	dic = {
		'job_name': 'worker',
		'task_index': 0,
		'batch_size': 128,
		'ps_hosts': ['192.168.0.101:22221'],
		'worker_hosts': ['192.168.0.103:22221', '192.168.0.102:22221'],
		'training_epochs': 5,
		'learning_rate': 1e-3,
		'train_steps': 1200
	}
	Distribute(**dic).main()
