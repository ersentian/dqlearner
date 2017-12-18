import sys
import matplotlib.pyplot as plt
from random import randrange

sys.path.append('Arcade-Learning-Environment/ale_python_interface')
print(sys.path)
from ale_python_interface import ALEInterface

def reduce_screen():
	pass

def train():
	ale = ALEInterface()
	ale.setInt('random_seed',123)
	ale.loadROM('roms/breakout.bin')
	legal_actions = ale.getLegalActionSet()
	total_reward = 0
	while not ale.game_over():
		a = legal_actions[randrange(len(legal_actions))]
		reward = ale.act(a)
		screen = None
		screen= ale.getScreenRGB()
		print(screen)
		plt.imshow(screen)
		plt.show()

		total_reward += reward
		print(total_reward)
	print('Episode end!')

if __name__=='__main__':
	train()