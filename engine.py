import numpy as np 
import sys, os
import time


class Maze():
	def __init__(self):
		# . -> empty, A -> apple, 1,2 -> player
		self.row_len = 16
		self.col_len = 21
		self.grid = np.asarray([['.' for j in range(self.col_len)] 
			for i in range(self.row_len)])


	def update(self, x, y, type):
		self.grid[x][y] = type


	def show(self):
		os.system('clear')
		for i in self.grid:
			for j in i:
				print j,
			print
		print 


class Timer():
	def __init__(self):
		self.reborn = False


	def time_move(self):
		if self.reborn is False:
			return
		self.reborn -= 1
		if self.reborn == 0:
			self.reborn = True
			return


class Apple(Timer):
	def __init__(self, x, y):
		Timer.__init__(self)
		self.x = x 
		self.y = y
		self.lock_time = 10


	def eaten(self):
		assert self.reborn == False
		self.reborn = self.lock_time 


class Player(Timer):
	def __init__(self, x, y, score):
		Timer.__init__(self)
		self.x = x 
		self.y = y
		self.score = score 
		self.lock_time = 20
		self.far_sight = 15
		self.near_sight = 10
		self.life_left = 2
		self.face = np.random.choice(['north', 'east', 'south', 'west'])
		self.reborn = False


	def get_sight(self):
		if self.face == 'north':
			return [self.x-self.far_sight, self.y-self.near_sight, 
				self.x+self.near_sight, self.y+self.near_sight]
		elif self.face == 'east':
			return [self.x-self.near_sight, self.y-self.near_sight, 
				self.x+self.near_sight, self.y+self.far_sight]
		elif self.face == 'south':
			return [self.x-self.near_sight, self.y-self.near_sight, 
				self.x+self.far_sight, self.y+self.near_sight]
		elif self.face == 'west':
			return [self.x-self.near_sight, self.y-self.far_sight, 
				self.x+self.near_sight, self.y+self.near_sight]


	def move(self, x, y):
		self.x, self.y = x, y 


	def rotate(self, face):
		self.face = face


	def get_hit(self):
		assert self.life_left > 0 and self.reborn == False
		self.life_left -= 1
		if self.life_left == 0:
			self.reborn = self.lock_time
		

	def eat_apple(self):
		self.score += 1


class Enviroment():
	def __init__(self):
		self.maze = Maze()
		self.players = []
		self.nb_apple = 80
		self.apples = [self.set_apple() for _ in range(self.nb_apple)]
		for name in ['1', '2']:
			self.players.append(self.set_player(name, 0))


	def set_player(self, name, score):
		x, y = self.cal_born_pos()
		self.maze.update(x, y, name)
		return Player(x, y, score)


	def set_apple(self):
		x, y = self.cal_born_pos()
		self.maze.update(x, y, 'A')
		return Apple(x, y)


	def cal_born_pos(self):
		# TODO: make sure there is a empty place 
		while True:
			ret_x = np.random.randint(self.maze.row_len)
			ret_y = np.random.randint(self.maze.col_len)
			if self.maze.grid[ret_x][ret_y] == '.':
				break
		return (ret_x, ret_y)


	def get_player_state(self, idx):
		if self.players[idx].reborn != False:
			return np.asarray([[]])
		x0, y0, x1, y1 = self.players[idx].get_sight()
		x0, y0 = self.legal_bound(x0, y0)
		x1, y1 = self.legal_bound(x1, y1)
		return self.maze.grid[x0:x1+1, y0:y1+1]


	def legal_bound(self, x, y):
		x = max(x, 0)
		x = min(x, self.maze.row_len-1)
		y = max(y, 0)
		y = min(y, self.maze.col_len-1)
		return (x, y)


	def cal_move_der(self, action, face):
		res = {
			('forward', 'north'): (-1, 0, face),
			('forward', 'south'): (1, 0, face),
			('forward', 'east'): (0, 1, face),
			('forward', 'west'): (0, -1, face),
			('left', 'north'): (0, -1, face),
			('left', 'south'): (0, 1, face),
			('left', 'east'): (-1, 0, face),
			('left', 'west'): (1, 0, face),
			('right', 'north'): (0, 1, face),
			('right', 'south'): (0, -1, face),
			('right', 'east'): (1, 0, face),
			('right', 'west'): (-1, 0, face),
			('backward', 'north'): (1, 0, face),
			('backward', 'south'): (-1, 0, face),
			('backward', 'east'): (0, -1, face),
			('backward', 'west'): (0, 1, face),
			('rotate_left', 'north'): (0, 0, 'west'),
			('rotate_left', 'east'): (0, 0, 'north'),
			('rotate_left', 'south'): (0, 0, 'east'),
			('rotate_left', 'west'): (0, 0, 'south'),
			('rotate_right', 'north'): (0, 0, 'east'),
			('rotate_right', 'east'): (0, 0, 'south'),
			('rotate_right', 'south'): (0, 0, 'west'),
			('rotate_right', 'west'): (0, 0, 'north')
		}
		return res.get((action, face), (0, 0, face))


	def judge_been_hit(self, face, x1, y1, x2, y2, reborn):
		if reborn != False:
			return False
		if face == 'north' and y2 == y1 and x2 < x1:
			return True
		if face == 'east' and x1 == x2 and y2 > y1:
			return True
		if face == 'south' and y1 == y2 and x2 > x2:
			return True
		if face == 'west' and x1 == x2 and y1 < y2:
			return True
		return False


	def time_move(self):
		self.players[0].time_move()
		self.players[1].time_move()
		if self.players[0].reborn == True:
			self.players[0] = self.set_player('1', self.players[0].score)
		if self.players[1].reborn == True:
			self.players[1] = self.set_player('2', self.players[1].score)
		for i in range(self.nb_apple):
			self.apples[i].time_move()
			if self.apples[i].reborn == True:
				self.apples[i] = self.set_apple()


	def remove_beam(self):
		for i in range(self.maze.row_len):
			for j in range(self.maze.col_len):
				if self.maze.grid[i][j] in ['^','>','<','v']:
					self.maze.update(i, j, '.')


	def beam_animation(self, x, y, face):
		if face == 'north':
			for i in range(x):
				if self.maze.grid[i][y] == '.':
					self.maze.update(i, y, '^')
		elif face == 'east':
			for i in range(y+1, self.maze.col_len):
				if self.maze.grid[x][i] == '.':
					self.maze.update(x, i, '>')
		elif face == 'west':
			for i in range(y):
				if self.maze.grid[x][i] == '.':
					self.maze.update(x, i, '<')
		elif face == 'south':
			for i in range(x+1, self.maze.row_len):
				if self.maze.grid[i][y] == '.':
					self.maze.update(i, y, 'v')


	def show_all(self):
		self.maze.show()
		print '         Player 1             Player 2'
		print 'Score      %d                     %d' % (self.players[0].score, self.players[1].score)
		print 'Life       %d                     %d' % (self.players[0].life_left, self.players[1].life_left)
		print 'Reborn    %5s                 %5s' % (self.players[0].reborn, self.players[1].reborn)


	def take_action(self, action1, action2, beam_animation=False):
		reward1, reward2 = 0, 0
		self.time_move()
		x1, y1, face1 = self.players[0].x, self.players[0].y, self.players[0].face
		x2, y2, face2 = self.players[1].x, self.players[1].y, self.players[1].face
		reborn1, reborn2 = self.players[0].reborn, self.players[1].reborn
		if reborn1 == False:
			dx, dy, face1 = self.cal_move_der(action1, face1)
			next_x1, next_y1 = x1+dx, y1+dy
			next_x1, next_y1 = self.legal_bound(next_x1, next_y1)
		else:
			next_x1, next_y1 = x1, y1 
		
		if reborn2 == False:
			dx, dy, face2 = self.cal_move_der(action2, face2)
			next_x2, next_y2 = x2+dx, y2+dy
			next_x2, next_y2 = self.legal_bound(next_x2, next_y2)
		else:
			next_x2, next_y2 = x2, y2

		# move colision
		if reborn1 == False and reborn2 == False and (next_x1, next_y1) == (next_x2, next_y2):
			return (self.get_player_state(0), 0, self.get_player_state(1), 0) 

		if reborn1 == False and action1 == 'beam' and self.judge_been_hit(
			face1, next_x1, next_y1, next_x2, next_y2, reborn2):
				self.players[1].get_hit()
				
		if reborn2 == False and action2 == 'beam' and self.judge_been_hit(
			face2, next_x2, next_y2, next_x1, next_y1, reborn1):
				self.players[0].get_hit()
				
		# update maze and players
		next_reborn1, next_reborn2 = self.players[0].reborn, self.players[1].reborn 
		if reborn1 == False and next_reborn1 == False:
			self.players[0].rotate(face1)
			self.players[0].move(next_x1, next_y1) 
			self.maze.update(x1, y1, '.')
			self.maze.update(next_x1, next_y1, '1')
		if reborn1 == False and next_reborn1 != False:
			self.maze.update(x1, y1, '.')

		if reborn2 == False and next_reborn2 == False:
			self.players[1].rotate(face2)
			self.players[1].move(next_x2, next_y2)
			if reborn1 == False and reborn2 == False and (x2, y2) == (next_x1, next_y1):
				pass
			else:
				self.maze.update(x2, y2, '.')
			self.maze.update(next_x2, next_y2, '2')
		if reborn2 == False and next_reborn2 != False:
			self.maze.update(x2, y2, '.')

		# update apples
		for i in range(self.nb_apple):
			if self.apples[i].reborn == False:
				if reborn1 == False and (self.apples[i].x, self.apples[i].y) == (next_x1, next_y1):
					self.apples[i].eaten()
					self.players[0].eat_apple()
					reward1 = 100
				if reborn2 == False and (self.apples[i].x, self.apples[i].y) == (next_x2, next_y2):
					self.apples[i].eaten()
					self.players[1].eat_apple()
					reward2 = 100

		# beam animation
		if beam_animation:
			if reborn1 == False and action1 == 'beam':
				self.beam_animation(x1, y1, face1)
			if reborn2 == False and action2 == 'beam':
				self.beam_animation(x2, y2, face2)
			self.show_all()
			self.remove_beam()

		return (self.get_player_state(0), reward1, self.get_player_state(1), reward2)


def random_action():
	return np.random.choice(['forward', 'left', 'right', 'backward', 
		'rotate_left', 'rotate_right', 'beam', 'stay'])


if __name__ == '__main__':
	env = Enviroment()
	env.maze.show()
	t = 100000
	while True:
		# time.sleep(1.)
		action1 = random_action()
		# action1 = raw_input()
		action2 = random_action()
		env.take_action(action1, action2)
		env.show_all() 
		t -= 1
		if t == 0:
			break