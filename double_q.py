import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import random
import time
import json
import gym
from gym import wrappers
from keras.models import load_model


def mlp():
	from keras.models import Sequential
	from keras.layers import Dense, Flatten, Input

	model = Sequential()
	model.add(Dense(32, activation='relu', input_shape=(500,)))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(6))
	model.compile(loss='mse', optimizer='RMSprop', metrics=['mae'])
	model.summary()

	return model 


def one_hot_encoder(n):
	ret = np.zeros(500)
	ret[n] = 1
	return ret


def replay(replay_memory, action_function, estimate_function, step):
	batch_size = 32
	gema = 0.99
	C_step = 10

	batch_data = random.sample(replay_memory, batch_size)
	states = np.asarray([raw[0] for raw in batch_data])
	next_states = np.asarray([raw[2] for raw in batch_data])
	scores = estimate_function.predict(states)
	next_scores = estimate_function.predict(next_states)
	action_scores = action_function.predict(next_states)
	y = []
	for raw, score_i, next_score_i, action_score_i in zip(
		batch_data, scores, next_scores, action_scores): 
		_, action, next_state, reward, done = raw
		idx = action
		best_next_action = np.argmax(action_score_i)
		best_next_score = next_score_i[best_next_action]
		# if done is True, we don't need to estimate the Q value of next state
		tar = reward + done * gema * best_next_score
		score_i[idx] = tar
		y.append(score_i)
	x = states
	y = np.asarray(y)
	
	loss_action = action_function.train_on_batch(x, y)[1]
	if step % C_step == 0:
		loss_estimate = estimate_function.train_on_batch(x, y)[1]
		# print 'Loss', loss_estimate, loss_action 


def plot(rewards, q):
	plt.plot(rewards, 'b', q, 'r')
	plt.savefig('models/index.jpg')


def train():
	from collections import deque

	action_function = mlp()
	estimate_function = mlp()
	# action_function = load_model('models/actionQ_7800_-2.22')
	# estimate_function = load_model('models/estimateQ_7800_-2.22')
	real_q = json.loads(open('Taxi_q').readline())
	env = gym.make("Taxi-v2")
	episode_reward_tracker = []
	episode_len_tracker = []
	ave_q_error = []
	max_q_error = []
	replay_memory = deque()
	max_replay_memory_size = 1000000
	min_replay_memory_size = 100000
	epsilon_start = 1
	epsilon_end = 0.1
	epsilon_decay_steps = 2000000
	epsilon_decay_val = (epsilon_start-epsilon_end) / epsilon_decay_steps
	epsilon = epsilon_start
	nb_episodes = 20000
	max_episode_steps = 200

	for t in range(nb_episodes):
		observation = env.reset()
		# env.render()
		whole_reward = 0
		for step in range(max_episode_steps):
			epsilon = max(epsilon_end, epsilon - epsilon_decay_val) 
			if np.random.random() < epsilon:
				action = env.action_space.sample()
			else:
				state = np.asarray([one_hot_encoder(observation)])
				q_values = action_function.predict(state)[0]
				standard_q = [real_q[str((observation, i))] for i in range(6)]
				right_action = np.argmax(standard_q)
				# print q_values
				# print standard_q # taxi real Q
				ave_q_error.append(sum([np.abs(q_values[i]-standard_q[i]) for i in range(6)]) / 6)
				max_q_error.append(np.abs(q_values[right_action]-standard_q[right_action]))
				action = np.argmax(q_values)
			next_observation, reward, done, _ = env.step(action)
			whole_reward += reward
			# env.render()
			if len(replay_memory) > max_replay_memory_size:
				replay_memory.popleft()
			replay_memory.append((one_hot_encoder(observation), action, 
				one_hot_encoder(next_observation), reward, 0 if done else 1))

			if len(replay_memory) > min_replay_memory_size:
				replay(replay_memory, action_function, estimate_function, step)
				
			if done is True:
				# Save model
				episode_reward_tracker.append(whole_reward)
				episode_len_tracker.append(step+1)
				print t, step, sum(ave_q_error[-200:]) / 200, sum(max_q_error[-200:]) / 200
				if t % 100 == 0:
					plot(episode_reward_tracker, episode_len_tracker)
					action_function.save('models/actionQ_%d_%.2lf' % (t, whole_reward))
					estimate_function.save('models/estimateQ_%d_%.2lf' % (t, whole_reward))
				break
			else:
				observation = next_observation


def play():
	env = gym.make('Taxi-v2')
	action_function = load_model('models/actionQ_15300_-0.76')
	estimate_function = load_model('models/estimateQ_15300_-0.76')
	nb_episodes = 20000
	max_episode_steps = 200
	epsilon = 0.05

	for t in range(nb_episodes):
		observation = env.reset()
		env.render()
		whole_reward = 0
		for step in range(max_episode_steps):
			if np.random.random() < epsilon:
				action = env.action_space.sample()
			else:
				state = np.asarray([one_hot_encoder(observation)])
				q_values = action_function.predict(state)[0]
				print q_values
				action = np.argmax(q_values)
			next_observation, reward, done, _ = env.step(action)
			whole_reward += reward
			env.render()
			time.sleep(1)
			if done is True:
				print whole_reward
				break
			else:
				observation = next_observation


if __name__ == '__main__':
	train()
	# play()