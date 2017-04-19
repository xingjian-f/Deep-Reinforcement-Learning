import numpy as np
import random 
import os
from collections import deque
from engine import Enviroment


def mlp():
	from keras.models import Sequential
	from keras.layers import Dense, Flatten, Input

	model = Sequential()
	model.add(Flatten(input_shape=(16, 21)))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(8))
	model.compile(loss='mse', optimizer='adam', metrics=['mae'])
	print model.summary()

	return model 


def random_action(all_actions):
	return np.random.choice(all_actions)


def trans_data(x, char_dict):
	assert isinstance(x, np.ndarray), x
	ret = []
	for i in range(16):
		ret.append([])
		for j in range(21):
			if i<x.shape[0] and j<x.shape[1]:
				ret[-1].append(x[i][j])
			else:
				ret[-1].append('empty')
	for i in range(16):
		for j in range(21):
			ret[i][j] = char_dict[ret[i][j]]
	return np.asarray([ret])


def train():
	from keras.models import load_model
	# q1 = mlp()
	q1 = load_model('models/q1.hdf5')
	q2 = mlp()
	env = Enviroment()
	all_actions = ['forward', 'left', 'right', 'backward', 
		'rotate_left', 'rotate_right', 'beam', 'stay']
	state1 = env.get_player_state(0)
	state2 = env.get_player_state(1)
	env.maze.show()
	char_dict = {'.':0, '1':1, '2':2, 'A':3, 'empty':4}
	t = int(1e7)
	epsilion = 1
	gema = 0.99
	learning_step = 0.5
	min_eps = 0.1
	decay_rate = 0.9 / 1e6
	sample_queue1 = deque()
	sample_queue2 = deque()
	queue_size = 1e6
	batch_size = 32
	for iteration in range(t):
		if np.random.random() < epsilion:
			action1 = random_action(all_actions)
		else:
			x = trans_data(state1, char_dict)
			pred_res = q1.predict(x)[0]
			action1 = all_actions[np.argmax(pred_res)]
		# if np.random.random() < epsilion:
		action2 = random_action(all_actions)
		# else:			
		# 	x = trans_data(state2, char_dict)
		# 	pred_res = q2.predict(x)[0]
		# 	action2 = all_actions[np.argmax(pred_res)]
 		next_state1, reward1, next_state2, reward2 = env.take_action(action1, action2)

 		sample_queue1.append((trans_data(state1, char_dict)[0], action1, 
 			reward1, trans_data(next_state1, char_dict)[0]))
 		sample_queue2.append((trans_data(state2, char_dict), action2,
 			reward2, trans_data(next_state2, char_dict)))
 		if len(sample_queue1) > queue_size:
 			sample_queue1.popleft()
 		if len(sample_queue2) > queue_size:
 			sample_queue2.popleft()
 		state1 = next_state1
 		state2 = next_state2
		epsilion = max(min_eps, epsilion - decay_rate)
		env.show_all()
		# train on batch
		if len(sample_queue1) >= batch_size:
			batch_data = random.sample(sample_queue1, batch_size)
			states = np.asarray([raw[0] for raw in batch_data])
			next_states = np.asarray([raw[-1] for raw in batch_data])
			scores = q1.predict(states)
			next_scores = q1.predict(next_states)
			x = []
			y = []
			for raw, score_i, next_score_i in zip(batch_data, scores, next_scores): 
				state, action, reward, next_state = raw
				idx = all_actions.index(action)
				best_next = max(next_score_i)
				tar = score_i[idx] + learning_step * (reward + gema*best_next - score_i[idx])
				score_i[idx] = tar
				x.append(state)
				y.append(score_i)
			x = np.asarray(x)
			y = np.asarray(y)
			loss = q1.train_on_batch(x, y)[1]
			print iteration, loss

		if iteration % 1000 == 0:
			q1.save('models/q1.hdf5')
		

def run():
	from keras.models import load_model
	q1 = load_model('models/q1.hdf5')
	env = Enviroment()
	eps = 0.1
	all_actions = ['forward', 'left', 'right', 'backward', 
		'rotate_left', 'rotate_right', 'beam', 'stay']
	state1 = env.get_player_state(0)
	state2 = env.get_player_state(1)
	env.maze.show()
	char_dict = {'.':0, '1':1, '2':2, 'A':3, 'empty':4}
	while True:
		if random.random() < eps:
			action1 = random_action(all_actions)
		else:
			x = trans_data(state1, char_dict)
			pred_res = q1.predict(x)[0]
			action1 = all_actions[np.argmax(pred_res)]
		action2 = random_action(all_actions)
 		next_state1, reward1, next_state2, reward2 = env.take_action(action1, action2, True)
 		state1 = next_state1
 		state2 = next_state2
		env.show_all()


if __name__ == '__main__':
	train()
	# run()