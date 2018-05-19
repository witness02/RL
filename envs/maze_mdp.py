# this is a self-environment for gym. you should copy this file to '/anaconda/envs/rl/lib/python3.4/
# site-packages/gym/envs/classic_control' directory, and modify file '.../classic_control/__init__.py'
# and '.../gym/envs/__init__.py'
import logging
from gym.envs.classic_control import rendering
import numpy as np
import gym

logger = logging.getLogger(__name__)


class MazeEnv(gym.Env):
	wall = 1
	door = 2
	action_space = ['n', 'e', 's', 'w']
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 2
	}
	
	def __init__(self):
		self.states = np.zeros(25, dtype=int).reshape(5, 5)
		self.walls = [(0, 3), (1, 3), (2, 0), (2, 1), (4, 2), (4, 3), (4, 4)]
		self.doors = [(2,4)]
		for w in self.walls:
			self.states[w] = self.wall
		for d in self.doors:
			self.states[d] = self.door
		
		self.nWidth = self.states.shape[1]
		self.nHeight = self.states.shape[0]
		
		self.state = (0, 0)
		
		self.viewer = None
	
	# return next state
	def state_transition(self, state, action):
		px = state[0]
		py = state[1]
		if 'n' == action:
			px -= 1
		elif 'e' == action:
			py += 1
		elif 's' == action:
			px += 1
		elif 'w' == action:
			px -= 1
		if -1 < px < self.states.shape[0] and -1 < py < self.states.shape[1] \
			and self.states[(px, py)] != self.wall:
			next_state = (px, py)
		else:
			next_state = state
		return next_state
	
	# return reward and is_terminal
	def state_reward(self, state):
		if self.door == self.states[state]:
			return 1, True
		return -1, False
	
	def _step(self, action):
		next_state = self.state_transition(self.state, action)
		reward, is_terminal = self.state_reward(next_state)
		self.state = next_state
		return next_state, reward, is_terminal, {}
	
	def _render(self, mode='human', close=False):
		print("render: ", self.state)
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return
		screen_width = 100 * (self.nWidth + 2)
		screen_height = 100 * (self.nHeight + 2)
		
		if self.viewer is None:
			self.viewer = rendering.Viewer(screen_width, screen_height)
			self.lines = dict()
			# draw horizontal lines
			for i in range(self.nHeight + 1):
				line_item = rendering.Line((100 * (i + 1), 100), (100 * (i + 1), 100 * (self.nWidth + 1)))
				self.viewer.add_geom(line_item)
			# draw vertical lines
			for i in range(self.nWidth + 1):
				line_item = rendering.Line((100, 100 * (i + 1)), (100 * (self.nHeight + 1), 100 * (i + 1)))
				self.viewer.add_geom(line_item)
			
			# draw walls
			for i in range(len(self.walls)):
				wall = self.walls[i]
				item = rendering.make_polygon([(0, 0), (0, 100), (100, 100), (100, 0)], True)
				item.set_color(0.9, 0.6, 0.1)
				self.circletrans = rendering.Transform(translation=(100 * (wall[1] + 1), 100 * (self.nHeight - wall[0])))
				item.add_attr(self.circletrans)
				self.viewer.add_geom(item)
			
			# draw doors
			for i in range(len(self.doors)):
				door = self.doors[i]
				item = rendering.make_polygon([(0, 0), (0, 100), (100, 100), (100, 0)], True)
				item.set_color(0.2, 0.9, 0.1)
				self.circletrans = rendering.Transform(translation=(100 * (door[1] + 1), 100 * (self.nHeight - door[0])))
				item.add_attr(self.circletrans)
				self.viewer.add_geom(item)
			
			# draw person
			self.person = rendering.make_polygon([(0, 0), (0, 100), (100, 100), (100, 0)], True)
			self.person.set_color(0, 0, 0)
			self.person_trans = rendering.Transform(translation=(100 * (self.state[1] + 1), 100 * (self.nHeight - self.state[0])))
			self.person.add_attr(self.person_trans)
			self.viewer.add_geom(self.person)
		
		self.person_trans.set_translation(100 * (self.state[1] + 1), 100 * (self.nHeight - self.state[0]))
		
		return self.viewer.render(return_rgb_array=mode == 'rgb_array')
	
	def _reset(self):
		self.state = (0, 2)
