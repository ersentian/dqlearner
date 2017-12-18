import tensorflow as tf

class TFAgent:
	def __init__(self,size,layer_specification):
		self.sess = tf.Session()
		self.state_placeholder = tf.placeholder(shape=size,dtype=tf.float32,name='State_placeholder')
		self.reward_placeholder = tf.placeholder(shape=(None,),dtype=tf.float32,name='Reward_placeholder')
		self.action_index = tf.placeholder(shape=(None,None),dtype=tf.int32,name='Action_taken')
		
		#Main variables and graph definition
		self.define_variables(layer_specification)
		self.action = self.define_graph(self.state_placeholder)

		#Define loss and optimizer
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
		
		self.q_val = tf.gather_nd(self.action,self.action_index)
		self.q_val = tf.reshape(self.q_val,shape=(-1,))

		self.loss = tf.losses.mean_squared_error(self.reward_placeholder,self.q_val)
		self.opt = self.optimizer.minimize(-1*self.loss)

		#Initiate varables
		init = tf.global_variables_initializer()
		self.sess.run(init)

		self.train_writer = tf.summary.FileWriter('results/',self.sess.graph)
		#Maybe add summaries later

		#Now the learner is instantiated and ready!

	def define_variables(self,layer_specification):
		self.variables = []
		self.biases = []
		self.fully_connected = []

		num_inputs = 1
		for i in range(len(layer_specification)):
			next_num_inputs = layer_specification[i]

			with tf.name_scope('Convolutional_weights'):
				self.variables.append(tf.Variable(tf.random_normal(shape=(17,17,1,1),stddev=0.5),dtype=tf.float32))
				self.biases.append(tf.Variable(tf.random_normal(shape=(1,),stddev=0.5),dtype=tf.float32))
			
			num_inputs = next_num_inputs

		with tf.name_scope('Fully_connected'):
			#for i in range(0,2):
			self.fully_connected.append(tf.Variable(tf.random_normal(shape=(29*29,15),stddev=0.5,dtype=tf.float32)))
			self.fully_connected.append(tf.Variable(tf.random_normal(shape=(15,3),stddev=0.5,dtype=tf.float32)))

		for var in self.variables:
			print(var)
		for fc in self.fully_connected:
			print(fc)

	def define_graph(self,state):
		for i,var in enumerate(self.variables):
			with tf.name_scope('Convolutional_layer'):
				state = tf.nn.conv2d(state,var,strides=[1,1,1,1],padding='VALID')
				state = tf.add(state,self.biases[i])

			with tf.name_scope('Pooling_layer'):
				state = tf.nn.pool(state,window_shape=[2,2],pooling_type='MAX',strides=[2,2],padding='SAME')
			
			with tf.name_scope('Activation_function'):
				state = tf.sigmoid(state)
			print(state)

		#print(state.get_shape().as_list())
		state = tf.reshape(state,shape=[-1,29*29])
		#print(state.get_shape().as_list())
		out = state

		for fc in self.fully_connected:
			out = tf.matmul(out,fc)
			#print(out.get_shape().as_list())
		return out

	def get_action(self,observation):
		return self.sess.run(self.action,feed_dict={self.state_placeholder: observation})

	def train_step(self,inputs_batch,targets_batch,actions_batch):
		self.sess.run(self.opt,feed_dict={self.state_placeholder: inputs_batch,self.reward_placeholder: targets_batch, self.action_index: actions_batch})
	
	def restore_session(self,i):
		saver = tf.train.Saver()
		saver.restore(self.sess,'results/models/model_%d.cpkt'%i)
		#print(loss)

		#self.define_conv_layer()
		#self.define_conv_graph(self.state)
		#self.define_fc_layer()
		#self.define_fc_graph(self.state)
