import os,sys,time,pickle

from skimage.measure import block_reduce
from skimage.transform import resize

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

#from tests import *

from dqagent import DQAgent
from rfagent import TFAgent

import pygame
pygame.init()

class Ballgame:
	def __init__(self,size):
		self.size = size #Board size
		self.board = np.zeros(size)

		self.paddle_location = [int(size[0]/2),0] #Paddle location
		self.half_paddle_length = 25
		self.paddle_thickness = 10

		self.ball_location = [int(np.random.normal(size[0]/2,scale=10)),int(np.random.normal(size[1]/2,scale=10))] #In the middle with an offset
		self.half_ball_length = 5
		
		#Initiate ball velocity
		x_magnitude = np.random.randint(1,3)
		y_magnitude = np.random.randint(1,3)
		x_sign = np.random.randint(2)
		if x_sign == 0:
			self.velocity = [x_magnitude,y_magnitude]
		else:
			self.velocity = [-x_magnitude,y_magnitude]
		
		#self.ball_steps = [0,0] #Count steps to get magnitude - No: don't!

		self.init_board()

	#Paint the board
	def init_board(self):
		### Add board ###
		self.board[(self.paddle_location[0]-self.half_paddle_length):(self.paddle_location[0]+self.half_paddle_length+1),\
		0:self.paddle_thickness] = np.ones((2*self.half_paddle_length+1,self.paddle_thickness))
		### Add ball ###
		self.board[(self.ball_location[0]-self.half_ball_length):(self.ball_location[0]+self.half_ball_length+1),\
		(self.ball_location[1]-self.half_ball_length):(self.ball_location[1]+self.half_ball_length+1)] = np.ones((2*self.half_ball_length+1,2*self.half_ball_length+1))

	def interpret(self,action):
		action_dict = ['left','wait','right']
		return action_dict[int(action)]

#	def get_state(self):
#		return np.expand_dims(np.expand_dims(self.board,axis=0),axis=3)

	def get_reduced_state(self):
		return resize(self.board,(84,84))

	def update_ball(self):
		#self.ball_steps[0] = self.ball_steps[0] + 1
		#self.ball_steps[1] = self.ball_steps[1] + 1
		
		### Remove old location ###
		self.board[(self.ball_location[0]-self.half_ball_length):(self.ball_location[0]+self.half_ball_length+1),\
		(self.ball_location[1]-self.half_ball_length):(self.ball_location[1]+self.half_ball_length+1)] = np.zeros((2*self.half_ball_length+1,2*self.half_ball_length+1))

		#Left and right
		self.ball_location[0] = self.ball_location[0] + self.velocity[0]
		if np.sign(self.velocity[0]) == 1:
			if self.ball_location[0] + self.half_ball_length >= (self.size[0]-1):
				self.ball_location[0] = self.size[0]-1-self.half_ball_length
				self.velocity[0] = -1*self.velocity[0]

		elif np.sign(self.velocity[0])==-1:
			if self.ball_location[0]-self.half_ball_length <= 0:
				self.ball_location[0] = self.half_ball_length
				self.velocity[0] = -1*self.velocity[0]

		#Up and down
		self.ball_location[1] = self.ball_location[1] + self.velocity[1]
		if np.sign(self.velocity[1])==1:
			if self.ball_location[1]+self.half_ball_length >= (self.size[1]-1):
				self.ball_location[1] = self.size[1]-1-self.half_ball_length
				self.velocity[1] = -1*self.velocity[1]
			
		elif np.sign(self.velocity[1])==-1:
			#Check for paddle
			ball_lower_bound = self.ball_location[1]-self.half_ball_length
			if (ball_lower_bound <= self.paddle_thickness) and (ball_lower_bound-self.velocity[1] > self.paddle_thickness):
				if (self.paddle_location[0]-self.half_paddle_length-self.half_ball_length)<self.ball_location[0]<(self.paddle_location[0]+self.half_paddle_length+self.half_ball_length):
					self.velocity[1] = -1*self.velocity[1]
					self.ball_location[1] = self.paddle_thickness+self.half_ball_length
					### Re-paint the ball ###
					self.board[(self.ball_location[0]-self.half_ball_length):(self.ball_location[0]+self.half_ball_length+1),\
					(self.ball_location[1]-self.half_ball_length):(self.ball_location[1]+self.half_ball_length+1)] = np.ones((2*self.half_ball_length+1,2*self.half_ball_length+1))	
					return 1
				else:
					pass
			elif self.ball_location[1]-self.half_ball_length <= 0:
				return -1 #You died!
		
		### Re-paint the ball ###
		self.board[(self.ball_location[0]-self.half_ball_length):(self.ball_location[0]+self.half_ball_length+1),\
		(self.ball_location[1]-self.half_ball_length):(self.ball_location[1]+self.half_ball_length+1)] = np.ones((2*self.half_ball_length+1,2*self.half_ball_length+1))
		return 0

	def update_state(self,action):
		paddle_speed = 2

		if action == 'left':
			self.board[self.paddle_location[0]-self.half_paddle_length:self.paddle_location[0]+self.half_paddle_length+1,0:self.paddle_thickness] = np.zeros((1,self.paddle_thickness))

			self.paddle_location[0] = self.paddle_location[0] - paddle_speed ### Move paddle ###
			if (self.paddle_location[0] - self.half_paddle_length) < 0:
				self.paddle_location[0] = self.half_paddle_length
			

			### Paint board ###
			self.board[self.paddle_location[0]-self.half_paddle_length:self.paddle_location[0]+self.half_paddle_length+1,0:self.paddle_thickness] = np.ones((1,self.paddle_thickness))
			return self.update_ball()

		elif action == 'right':
			self.board[self.paddle_location[0]-self.half_paddle_length:self.paddle_location[0]+self.half_paddle_length+1,0:self.paddle_thickness] = np.zeros((1,self.paddle_thickness))

			self.paddle_location[0] = self.paddle_location[0] + paddle_speed ### Move paddle ###
			if (self.paddle_location[0] + self.half_paddle_length) >= self.board.shape[0]:
				self.paddle_location[0] = self.board.shape[0]-self.half_paddle_length-1
			
			### Paint board ###
			self.board[self.paddle_location[0]-self.half_paddle_length:self.paddle_location[0]+self.half_paddle_length+1,0:self.paddle_thickness] = np.ones((1,self.paddle_thickness))
			return self.update_ball()

		elif action =='wait':
			return self.update_ball()