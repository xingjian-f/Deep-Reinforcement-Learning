from architectures import policy_network, dueling_network_ave
from util import one_hot_encoder, plot, p_hot_encoder
from keras.models import load_model
import numpy as np
import json
from collections import deque
import gym
import random 


def replay(replay_memory, actor, critic):
	batch_size = 32
	gema = 0.99

	batch_data = random.sample(replay_memory, batch_size)
	states = np.asarray([raw[0] for raw in batch_data])
	next_states = np.asarray([raw[2] for raw in batch_data])
	scores = critic.predict(states)
	next_scores = critic.predict(next_states)
	probs = actor.predict(next_states)
	
	y_q = []
	y_r = scores
	for raw, score_i, next_score_i, prob_i in zip(
		batch_data, scores, next_scores, probs): 
		_, action, next_state, reward, done = raw
		idx = action
		expected_reward = sum(next_score_i * prob_i)
		# if done is True, we don't need to estimate the Q value of next state
		tar = reward + done * gema * expected_reward
		score_i[idx] = tar
		y_q.append(score_i)
	x = states
	y_q = np.asarray(y_q)
	y_r = (y_r - np.mean(y_r)) / (np.std(y_r) * 1000 )
	
	actor.train_on_batch(x, y_r)
	loss_critic = critic.train_on_batch(x, y_q)[1]
	# print 'Loss', loss_critic 


def update_memory(replay_memory, max_replay_memory_size, observation, next_observation,
	state_size, action, reward, done):
	if len(replay_memory) > max_replay_memory_size:
		replay_memory.popleft()
	replay_memory.append((one_hot_encoder(observation, state_size), action, 
		one_hot_encoder(next_observation, state_size), reward, 0 if done else 1))


def select_action(policy_net, state):
	probs = policy_net.predict(np.asarray([state]))[0]
	# print list(state).index(1)
	# print probs
	action = np.random.choice(len(probs), p=probs)
	return action 


def train():
	# game info
	game_name = 'Taxi-v2'
	env = gym.make(game_name)
	state_size = env.observation_space.n 
	action_size = env.action_space.n
	print 'State size', state_size
	print 'Action size', action_size
	actor = policy_network(state_size, action_size)
	# actor = load_model('models/policy_11000_8.00')
	critic = dueling_network_ave(state_size, action_size)

	replay_memory = deque()
	max_replay_memory_size = 200000
	min_replay_memory_size = 100000
	episode_reward_tracker = []
	episode_len_tracker = []
	nb_episodes = 200000
	max_episode_steps = 200
	model_save_freq = 500
	update_freq = 10
	gema = 0.99

	for t in xrange(nb_episodes):
		observation = env.reset()
		whole_reward = 0
		for step in xrange(max_episode_steps):
			state = one_hot_encoder(observation, state_size)
			action = select_action(actor, state)
			next_observation, reward, done, _ = env.step(action)
			update_memory(replay_memory, max_replay_memory_size, observation, next_observation,
				state_size, action, reward, done)
			whole_reward += reward

			if len(replay_memory) > min_replay_memory_size and step % update_freq == 0:
				replay(replay_memory, actor, critic)
			if done:
				# Save model
				episode_reward_tracker.append(whole_reward)
				episode_len_tracker.append(step+1)
				ave_reward = float(sum(episode_reward_tracker[-100:])) / 100
				ave_len = float(sum(episode_len_tracker[-100:])) / 100
				print t, step, ave_reward, ave_len, whole_reward
				if len(replay_memory) > min_replay_memory_size and t % model_save_freq == 0:
					plot(episode_reward_tracker, episode_len_tracker)
					actor.save('models/actor_%d_%.2lf' % (t, ave_reward))
					critic.save('models/critic_%d_%.2lf' % (t, ave_reward))
				break
				
			else:
				observation = next_observation

if __name__ == '__main__':
	train()