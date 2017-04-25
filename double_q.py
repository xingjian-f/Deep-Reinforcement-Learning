from collections import deque
import random
import time
import json
import numpy as np 
import gym
from gym import wrappers
from keras.models import load_model
from architectures import mlp, dueling_network_naive, dueling_network_max, dueling_network_ave
from util import one_hot_encoder, plot


# @profile
def replay(replay_memory, action_function, estimate_function, update_estimate):
	batch_size = 32
	gema = 0.99

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
	if update_estimate:
		loss_estimate = estimate_function.train_on_batch(x, y)[1]
		# print 'Loss', loss_estimate, loss_action 


def e_greedy(epsilon, env, observation, state_size, action_function):
	if np.random.random() < epsilon:
		action = env.action_space.sample()
	else:
		state = np.asarray([one_hot_encoder(observation, state_size)])
		q_values = action_function.predict(state)[0]
		# print q_values
		action = np.argmax(q_values)
	return action 


def update_memory(replay_memory, max_replay_memory_size, observation, next_observation,
	state_size, action, reward, done):
	if len(replay_memory) > max_replay_memory_size:
		replay_memory.popleft()
	replay_memory.append((one_hot_encoder(observation, state_size), action, 
		one_hot_encoder(next_observation, state_size), reward, 0 if done else 1))

# @profile
def train():
	# game info
	game_name = 'Taxi-v2'
	env = gym.make(game_name)
	state_size = env.observation_space.n 
	action_size = env.action_space.n
	print 'State size', state_size
	print 'Action size', action_size

	# initialize deep model
	# action_function = mlp(state_size, action_size)
	# estimate_function = mlp(state_size, action_size)
	# action_function = dueling_network_ave(state_size, action_size)
	# estimate_function = dueling_network_ave(state_size, action_size)
	action_function = dueling_network_max(state_size, action_size)
	estimate_function = dueling_network_max(state_size, action_size)
	# action_function = load_model('models/actionQ_29000_2.00')
	# estimate_function = load_model('models/estimateQ_29000_2.00')

	# set the hyper-parameters 
	episode_reward_tracker = []
	episode_len_tracker = []
	replay_memory = deque()
	max_replay_memory_size = 200000
	min_replay_memory_size = 100000
	epsilon_start = 1
	epsilon_end = 0.1
	epsilon_decay_steps = 1000000
	epsilon_decay_val = (epsilon_start-epsilon_end) / epsilon_decay_steps
	epsilon = epsilon_start
	nb_episodes = 200000
	max_episode_steps = 200
	action_update_freq = 4
	estimate_update_freq = 20
	model_save_freq = 500

	for t in range(nb_episodes):
		observation = env.reset()
		whole_reward = 0
		for step in range(max_episode_steps):
			epsilon = max(epsilon_end, epsilon - epsilon_decay_val) 
			action = e_greedy(epsilon, env, observation, state_size, action_function) # select action by e-greedy
			next_observation, reward, done, _ = env.step(action) # take action
			whole_reward += reward
			update_memory(replay_memory, max_replay_memory_size, observation, next_observation,
				state_size, action, reward, done)
			if len(replay_memory) > min_replay_memory_size:
				if (step % action_update_freq) == 0:
					update_estimate = (step % estimate_update_freq) == 0
					replay(replay_memory, action_function, estimate_function, update_estimate)
				
			if done is True:
				# Save model
				episode_reward_tracker.append(whole_reward)
				episode_len_tracker.append(step+1)
				ave_reward = sum(episode_reward_tracker[-100:]) / 100
				ave_len = sum(episode_len_tracker[-100:]) / 100
				print t, step, epsilon, ave_reward, ave_len, whole_reward 
				if t % model_save_freq == 0 and len(replay_memory) > min_replay_memory_size:
					plot(episode_reward_tracker, episode_len_tracker)
					action_function.save('models/actionQ_%d_%.2lf' % (t, ave_reward))
					estimate_function.save('models/estimateQ_%d_%.2lf' % (t, ave_reward))
				break
			else:
				observation = next_observation


def play():
	game_name = 'Taxi-v2'
	env = gym.make(game_name)
	state_size = env.observation_space.n 
	action_function = load_model('models/actionQ_12000_-1.00')
	estimate_function = load_model('models/estimateQ_12000_-1.00')
	nb_episodes = 100
	max_episode_steps = 200
	epsilon = 0.05

	for t in range(nb_episodes):
		observation = env.reset()
		env.render()
		whole_reward = 0
		for step in range(max_episode_steps):
			action = e_greedy(epsilon, env, observation, state_size, action_function) # select action by e-greedy
			next_observation, reward, done, _ = env.step(action)
			whole_reward += reward
			env.render()
			# print observation, action, reward
			time.sleep(1.0)
			if done is True:
				print 'Whole reward', whole_reward
				break
			else:
				observation = next_observation


if __name__ == '__main__':
	train()
	# play()