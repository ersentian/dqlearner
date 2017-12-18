import os,sys,time,pickle,time

from skimage.measure import block_reduce
from scipy.misc import imresize

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

import threading
#from tests import *
from game import Ballgame

from dqagent import DQAgent
from rfagent import TFAgent

class Data_holder():
	def __init__(self):
		N = 250000
		self.state_memory = np.zeros((84,84,N))
		self.new_state_memory = np.zeros((84,84,N))
		self.action_reward_memory = np.zeros((2,N))

	def get_train_batch(dqlearner,gamma,active_index,inputs_batch,rewards_batch,actions_batch,outputs_batch):
		pass

#Saving and loading
def save_game(path,obj):
	pickle.dump(obj,open(path,'wb'))
	return
def load_game(path):
	return pickle.load(open(path,'rb'))

### Training helper functions ###
def eps_greedy_choice(action,epsilon):
	num = np.random.uniform()
	if num < epsilon:
		action = np.random.randint(0,3)
	else:
		action = np.argmax(action[0])
	return action

#Merge reduced states to an input state
def merge_states(dq_state,states):
	for i,state in enumerate(states):
		dq_state[0,:,:,i] = state
	return

#Enter new state to the list of previous states
def enter_to_prev(state,previous_states):
	previous_states.append(state)
	del previous_states[0]
	return previous_states

### This requres overhead - most expensive function ###
def get_train_batch(dqlearner,gamma,active_index,state_memory,new_state_memory,action_reward_memory,inputs_batch,rewards_batch,actions_batch,outputs_batch):
	inxes = np.random.randint(0,active_index,size=actions_batch.shape[0])
	
	inputs_batch = state_memory[inxes,:,:,:]
	outputs_batch = new_state_memory[inxes,:,:,:]
	actions_batch[:,1] = action_reward_memory[inxes,0]
	

	q_hat = dqlearner.get_action(outputs_batch)
	print(dqlearner.get_action(inputs_batch))
	q_val = dqlearner.get_actions_taken(inputs_batch,actions_batch)
	print(q_val)
	print(q_val.shape)

	actions = np.argmax(q_hat,axis=1)

	for i in range(q_hat.shape[0]):
		if action_reward_memory[inxes[i],1] == -1:
			rewards_batch[i] = -1
		else:
			rewards_batch[i] = action_reward_memory[inxes[i],1]
			rewards_batch[i] += gamma*q_hat[i,actions[i]]
	return 

### Later on: Implement threading
class ThreadRunner:
	def __init__(self,dq_learner,n_threads,active_index,state_memory,new_state_memory,action_reward_memory):
		self.threads = []
		self.current_thread = 0
		self.dqlearner = dq_learner
		for i in range(n_threads):
			self.threads.append(threading.Thread(target=self.get_train_batch,args=(active_index,state_memory,new_state_memory,action_reward_memory)))
			self.threads[i].start()

	def get_train_batch(self,active_index,state_memory,new_state_memory,action_reward_memory):
		batch_size = 32
		actions_batch = np.zeros((batch_size,2)) ############
		actions_batch[:,0] = np.array(range(0,batch_size))
		rewards_batch = np.zeros(batch_size)

		inxes = np.random.randint(3,active_index,size=actions_batch.shape[0])
		total_inxes = [range(k-3,k+1) for k in inxes] 
		inputs_batch = np.swapaxes(np.swapaxes(state_memory[:,:,total_inxes],0,2),1,2)
		outputs_batch = np.swapaxes(np.swapaxes(new_state_memory[:,:,total_inxes],0,2),1,2)

		actions_batch[:,1] = action_reward_memory[0,inxes]

		rewards_batch[:] = action_reward_memory[1,inxes]
		q_hat = self.dqlearner.get_action(outputs_batch)
		actions = np.argmax(q_hat,axis=1)

		for i in range(0,q_hat.shape[0]):
			rewards_batch[i] += q_hat[i,actions[i]]
		return inputs_batch,actions_batch,rewards_batch

	def get_batch(self,active_index,state_memory,new_state_memory,action_reward_memory):
		print(self.threads)
		inputs_batch,actions_batch,rewards_batch = self.threads[self.current_thread].join()
		self.threads[current_thread] = threading.Thread(target=self.get_train_batch,args=(active_index,state_memory,new_state_memory,action_reward_memory))
		self.threads[current_thread].start()
		self.current_thread += 1
		self.current_thread = self.current_thread % self.n_threads

		return inputs_batch,actions_batch,rewards_batch

def train_dq():
	size = (210,160)
	game = Ballgame(size)

	dqlearner = DQAgent()

	###### Training hyperparameters ######
	N = 200000 #Size of replay memory, in samples
	batch_size = 32 #Mini-batch size
	train_steps = 10000000 #Number of training steps
	epsilon = 1 #Epsilon-greedy, annealed 1-> 0.1
	k = 4 #Factor that determines how often we make a decision
	gamma = 0.95 #Discount factor
	annealing_steps = 1000000
	replay_memory = [] #List of len max N
	C = 4
	####################################

	### Replay memory ###
	state_memory = np.zeros((N,84,84,4))
	new_state_memory = np.zeros((N,84,84,4))
	action_reward_memory = np.zeros((N,2))

	### Placeholders for memory reasons ###
	inputs_batch = np.zeros((batch_size,84,84,4))
	outputs_batch = np.zeros((batch_size,84,84,4))
	rewards_batch = np.zeros(batch_size).astype(np.float32)
	
	actions_batch = np.zeros((batch_size,2))
	actions_batch[:,0] = np.array(range(batch_size))


	dq_input_state = np.zeros((1,84,84,4))

	replay_index = 0
	active_index = 0
	### Initiate game, initiate first state ###
	previous_states = [] #Last 4 previous states 
	for i in range(4): ### Init
		state = game.get_reduced_state()
		previous_states.append(state)
		game.update_state('wait')

#	for i,state in enumerate(previous_states):
#		if i > 0:
#			assert not np.array_equal(previous_states[i],previous_states[i-1])

	merge_states(dq_input_state,previous_states)
	state_memory[replay_index,:,:,:] = dq_input_state[0,:,:,:]

	game.update_state('wait')	
	state = game.get_reduced_state()
	enter_to_prev(state,previous_states)
	merge_states(dq_input_state,previous_states)
	new_state_memory[replay_index,:,:,:] = dq_input_state[0,:,:,:]
	action_reward_memory[replay_index,:] = np.array([1,0])

	replay_index += 1
	active_index += 1

	game_len = 0
	game_lens = []
	for i in range(train_steps):
		game_len += 1

		#Save current state
		merge_states(dq_input_state,previous_states) #Merge previous 4 saved states to an input
		state_memory[replay_index,:,:,:] = dq_input_state[0,:,:,:]

		assert np.array_equal(state_memory[replay_index,:,:,:],new_state_memory[(replay_index-1),:,:,:])

		if i % k == 0: ### Take a Q-learner inspired step ###
			#Determine what action to take
			if i < annealing_steps:
				epsilon = 1 - (i/annealing_steps)*0.9

			action = dqlearner.get_action(dq_input_state)
			db_action = action #Save for debugging
			action = eps_greedy_choice(action,epsilon) #We can anneal epsilon

			#Take action and update game
			reward = game.update_state(game.interpret(action))

			### Add result to replay memory
			enter_to_prev(game.get_reduced_state(),previous_states)
			merge_states(dq_input_state,previous_states)
			new_state_memory[replay_index,:,:,:] = dq_input_state[0,:,:,:]
			action_reward_memory[replay_index,:] = np.array([action,reward])

			#Update indices
			replay_index += 1
			active_index += 1
			if replay_index >= N:
				replay_index = 0
			if active_index > N:
				active_index = N

			#### Something for debugging #######
			#if int(i/k) % 100 == 0: #DEBUGGING
			print('Training step %d'%(int(i/k)))
			print(epsilon)
			print(db_action[0])
			print(action)
			####################################

			get_train_batch(dqlearner,gamma,active_index,state_memory,new_state_memory,action_reward_memory,inputs_batch,rewards_batch,actions_batch,outputs_batch)
			dqlearner.train_step(inputs_batch,rewards_batch,actions_batch)

		else: ### Use the same action ###
			reward = game.update_state(game.interpret(action))

			enter_to_prev(game.get_reduced_state(),previous_states)
			merge_states(dq_input_state,previous_states)
			new_state_memory[replay_index,:,:,:] = dq_input_state[0,:,:,:]
			action_reward_memory[replay_index,:] = np.array([action,reward])

			replay_index += 1
			active_index += 1
			if replay_index >= N:
				replay_index = 0
			if active_index > N:
				active_index = N

		### Game over ###
		if reward == -1:
			print('Game over!')
			game_lens.append(game_len)
			g = dqlearner.sess.run(dqlearner.game_len_summary,feed_dict={dqlearner.game_len_placeholder:game_len})
			dqlearner.train_writer.add_summary(g,i)
			

			# Restart game 
			game = Ballgame(size)
			game_len = 0

			previous_states = []
			for i in range(4): ### Init
				state = game.get_reduced_state()
				previous_states.append(state)
				game.update_state('wait')

			for i,state in enumerate(previous_states):
				if i > 0:
					assert not np.array_equal(previous_states[i],previous_states[i-1])

			merge_states(dq_input_state,previous_states)
			state_memory[replay_index,:,:,:] = dq_input_state[0,:,:,:]

			game.update_state('wait')	
			state = game.get_reduced_state()
			enter_to_prev(state,previous_states)
			merge_states(dq_input_state,previous_states)

			new_state_memory[replay_index,:,:,:] = dq_input_state[0,:,:,:]
			action_reward_memory[replay_index,:] = np.array([1,0])
			#Game properly initiated
			replay_index += 1
			active_index += 1

		if i % 100000 == 0:
			saver = tf.train.Saver()
			saver.save(dqlearner.sess,'results/dqmodels/model_%d.cpkt'%i)

if __name__=='__main__':
	train_dq()
	#show_gameplay(2000000,'DQ')

	#test_dq_state()
	#play_game()
	
