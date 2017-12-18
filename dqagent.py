import tensorflow as tf

class DQAgent:
	def __init__(self):
		#Define session and placeholders
		self.sess = tf.Session()

		self.state_placeholder = tf.placeholder(shape=(None,84,84,4),dtype=tf.float32,name='State_placeholder')
		self.reward_placeholder = tf.placeholder(shape=(None,),dtype=tf.float32,name='Reward_placeholder')
		self.action_index = tf.placeholder(shape=(None,2),dtype=tf.int32,name='Action_taken')
		
		#Main variables and graph definition
		self.define_variables()
		self.action = self.define_graph(self.state_placeholder)

		print(self.action.get_shape())
		#Define loss and optimizer
		#self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
		self.optimizer = tf.train.RMSPropOptimizer(learning_rate=2.5e-7)

		self.q_val = tf.gather_nd(self.action,self.action_index)
		print(self.q_val.get_shape())
		self.q_val = tf.reshape(self.q_val,shape=(-1,))

		with tf.name_scope('Loss'):
			self.loss = tf.losses.mean_squared_error(self.reward_placeholder,self.q_val)
			self.opt = self.optimizer.minimize(-1*self.loss)

		#Initiate varables
		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.train_writer = tf.summary.FileWriter('results/',self.sess.graph)
		self.game_len_placeholder = tf.placeholder(shape=(),dtype=tf.int32,name='Game_length')
		self.game_len_summary = tf.summary.scalar('Game_length',self.game_len_placeholder)


	def define_variables(self):
		self.variables = []
		self.biases = []
		self.fully_connected = []

		with tf.name_scope('Convolutional_weights'):
			self.variables.append(tf.Variable(tf.random_normal(shape=(8,8,4,32),mean=0,stddev=0.02,dtype=tf.float32),name='conv_1'))
			self.variables.append(tf.Variable(tf.random_normal(shape=(4,4,32,64),mean=0,stddev=0.02,dtype=tf.float32),name='conv_2'))

			self.biases.append(tf.Variable(tf.random_normal(shape=(20,20,32),mean=0.05,stddev=0.02),dtype=tf.float32))
			self.biases.append(tf.Variable(tf.random_normal(shape=(4,4,64),mean=0.05,stddev=0.02),dtype=tf.float32))
		
		with tf.name_scope('Fully_connected_weights'):
			self.fully_connected.append(tf.Variable(tf.random_normal(shape=(256,256),mean=0.05,stddev=0.02,dtype=tf.float32),name='fc_1'))
			self.fully_connected.append(tf.Variable(tf.random_normal(shape=(256,3),mean=0.05,stddev=0.02,dtype=tf.float32),name='fc_2'))

			self.biases.append(tf.Variable(tf.random_normal(shape=(256,),mean=0.05,stddev=0.02),dtype=tf.float32))
		#for var in self.variables:
		#	print(var)
		#for fc in self.fully_connected:
		#	print(fc)

	def define_graph(self,state):
		with tf.name_scope('Convolutional_layer'):
			state = tf.nn.conv2d(state,self.variables[0],strides=[1,4,4,1],padding='VALID')
			state = tf.add(state,self.biases[0])
			#state = tf.nn.relu(state,name='Activation_function')
			state = tf.nn.max_pool(state,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
			state = tf.nn.relu(state,name='Activation_function')
			print(state)
		
		with tf.name_scope('Convolutional_layer'):
			state = tf.nn.conv2d(state,self.variables[1],strides=[1,2,2,1],padding='VALID')
			state = tf.add(state,self.biases[1])
			#state = tf.nn.relu(state,name='Activation_function')
			state = tf.nn.max_pool(state,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
			state = tf.nn.relu(state,name='Activation_function')
			print(state)

		#print('State after convolutional layers')
		#print(state.get_shape().as_list())
		
		#print(state.get_shape().as_list())
		state = tf.reshape(state,shape=[-1,256])
		with tf.name_scope('Fully_connected_layer'):
			out = state
			out = tf.matmul(out,self.fully_connected[0])
			out = tf.add(out,self.biases[2])
			out = tf.nn.relu(out,name='ReLU_activation')
			out = tf.matmul(out,self.fully_connected[1])
		return out

	def get_action(self,observation):
		return self.sess.run(self.action,feed_dict={self.state_placeholder: observation})

	def get_actions_taken(self,observation,actions):
		return self.sess.run(self.q_val,feed_dict={self.state_placeholder:observation, self.action_index: actions})

	def train_step(self,inputs_batch,targets_batch,actions_batch):
		self.sess.run(self.opt,feed_dict={self.state_placeholder: inputs_batch,self.reward_placeholder: targets_batch, self.action_index: actions_batch})
	
	def restore_session(self,steps):
		saver = tf.train.Saver()
		saver.restore(self.sess,'results/dqmodels/model_%d.cpkt'%steps)

class StaleAgent:
	def __init__(self):
		#Define session and placeholders
		self.sess = tf.Session()

		self.state_placeholder = tf.placeholder(shape=(None,84,84,4),dtype=tf.float32,name='State_placeholder')
		self.reward_placeholder = tf.placeholder(shape=(None,),dtype=tf.float32,name='Reward_placeholder')
		self.action_index = tf.placeholder(shape=(None,None),dtype=tf.int32,name='Action_taken')
		
		#Main variables and graph definition
		self.define_variables()
		self.action = self.define_graph(self.state_placeholder)

		#Define loss and optimizer
		#self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
		self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00001)

		self.q_val = tf.gather_nd(self.action,self.action_index)
		self.q_val = tf.reshape(self.q_val,shape=(-1,))

		with tf.name_scope('Loss'):
			self.loss = tf.losses.mean_squared_error(self.reward_placeholder,self.q_val)
			self.opt = self.optimizer.minimize(-1*self.loss)

	def transfer_values(self,dqlearner):
		pass

	def define_variables(self):
		self.variables = []
		self.biases = []
		self.fully_connected = []

		with tf.name_scope('Convolutional_weights'):
			self.variables.append(tf.Variable(tf.random_normal(shape=(8,8,4,32),mean=0,stddev=0.001,dtype=tf.float32),name='conv_1'))
			self.variables.append(tf.Variable(tf.random_normal(shape=(4,4,32,64),mean=0,stddev=0.001,dtype=tf.float32),name='conv_2'))
			#self.biases.append(tf.Variable(tf.random_normal(shape=(20,20,16),mean=1.0,stddev=0.5),dtype=tf.float32))
			#self.biases.append(tf.Variable(tf.random_normal(shape=(1,),mean=1.0,stddev=0.5),dtype=tf.float32))
		with tf.name_scope('Fully_connected_weights'):
			self.fully_connected.append(tf.Variable(tf.random_normal(shape=(256,256),mean=0,stddev=0.001,dtype=tf.float32),name='fc_1'))
			self.fully_connected.append(tf.Variable(tf.random_normal(shape=(256,3),mean=0,stddev=0.001,dtype=tf.float32),name='fc_2'))

		for var in self.variables:
			print(var)
		for fc in self.fully_connected:
			print(fc)

	def define_graph(self,state):
		with tf.name_scope('Convolutional_layer'):
			state = tf.nn.conv2d(state,self.variables[0],strides=[1,4,4,1],padding='VALID')
			#state = tf.add(state,self.biases[0])
			#state = tf.nn.relu(state,name='Activation_function')
			state = tf.nn.max_pool(state,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
			state = tf.nn.relu(state,name='Activation_function')
			print(state)
		
		with tf.name_scope('Convolutional_layer'):
			state = tf.nn.conv2d(state,self.variables[1],strides=[1,2,2,1],padding='VALID')
			#state = tf.add(state,self.biases[1])
			#state = tf.nn.relu(state,name='Activation_function')
			state = tf.nn.max_pool(state,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
			state = tf.nn.relu(state,name='Activation_function')
			print(state)

		print('State after convolutional layers')
		print(state.get_shape().as_list())
		state = tf.reshape(state,shape=[-1,256])
		print(state.get_shape().as_list())

		print('State before fully convolutional layers')
		with tf.name_scope('Fully_connected_layer'):
			out = state
			out = tf.matmul(out,self.fully_connected[0])
			out = tf.nn.relu(out,name='ReLU_activation')
			out = tf.matmul(out,self.fully_connected[1])
		return out

	def get_action(self,observation):
		return self.sess.run(self.action,feed_dict={self.state_placeholder: observation})

	def train_step(self,inputs_batch,targets_batch,actions_batch):
		self.sess.run(self.opt,feed_dict={self.state_placeholder: inputs_batch,self.reward_placeholder: targets_batch, self.action_index: actions_batch})
	
	def restore_session(self,steps):
		saver = tf.train.Saver()
		saver.restore(self.sess,'results/dqmodels/model_%d.cpkt'%steps)