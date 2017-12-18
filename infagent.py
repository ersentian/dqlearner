import tensorflow as tf

class InfAgent: #Informed agent
	def __init__(self):
		#Define session and placeholders
		self.sess = tf.Session()

		#Ball position, ball velocity, location of paddle, width of paddle
		self.state_placeholder = tf.placeholder(shape=(None,10),dtype=tf.float32,name='State_placeholder')
		self.reward_placeholder = tf.placeholder(shape=(None,),dtype=tf.float32,name='Reward_placeholder')
		self.action_index = tf.placeholder(shape=(None,None),dtype=tf.int32,name='Action_taken')
		
		#Main variables and graph definition
		self.define_variables()
		self.action = self.define_graph(self.state_placeholder)

		#Define loss and optimizer
		#self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
		self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.000001)

		self.q_val = tf.gather_nd(self.action,self.action_index)
		self.q_val = tf.reshape(self.q_val,shape=(-1,))

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

		with tf.name_scope('Fully_connected'):
			self.fully_connected.append(tf.Variable(tf.random_normal(shape=(10,10),mean=0,stddev=0.01,dtype=tf.float32)))
			self.fully_connected.append(tf.Variable(tf.random_normal(shape=(10,10),mean=0,stddev=0.01,dtype=tf.float32)))

		for var in self.variables:
			print(var)
		for fc in self.fully_connected:
			print(fc)

	def define_graph(self,state)
		out = tf.matmul(out,self.fully_connected[0])
		out = tf.nn.relu(state,name='ReLU_activation')
		out = tf.matmul(out,self.fully_connected[1])
		return out

	def get_action(self,observation):
		return self.sess.run(self.action,feed_dict={self.state_placeholder: observation})

	def train_step(self,inputs_batch,targets_batch,actions_batch):
		self.sess.run(self.opt,feed_dict={self.state_placeholder: inputs_batch,self.reward_placeholder: targets_batch, self.action_index: actions_batch})
	
	def restore_session(self,steps):
		saver = tf.train.Saver()
		saver.restore(self.sess,'results/dqmodels/model_%d.cpkt'%steps)
