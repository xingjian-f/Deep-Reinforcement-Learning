from architectures import policy_network
from util import one_hot_encoder, plot, p_hot_encoder
from keras.models import load_model
import numpy as np
import json
import gym 


def select_action(policy_net, state):
	probs = policy_net.predict(np.asarray([state]))[0]
	# print list(state).index(1)
	# print probs
	action = np.random.choice(len(probs), p=probs)
	return action 


def decay_rewards(rewards, gema):
	total_reward = 0
	ret = []
	for r in rewards[::-1]:
		total_reward *= gema
		total_reward += r 
		ret.append(total_reward)
	return ret[::-1]


def train(policy_net, queue, action_size):
	x = []
	y = []

	rewards = []
	states = []
	actions = []
	for i in queue:
		states.extend(i[0])
		actions.extend(i[1])
		rewards.extend(i[2])
	rewards = (rewards - np.mean(rewards)) / np.std(rewards)
	

	for i in range(len(states)):
		x.append(states[i])
		label = p_hot_encoder(actions[i], action_size, rewards[i])
		y.append(label)

	x = np.asarray(x)
	y = np.asarray(y)

	policy_net.train_on_batch(x, y) 


def run():
	# game info
	game_name = 'Taxi-v2'
	env = gym.make(game_name)
	state_size = env.observation_space.n 
	action_size = env.action_space.n
	print 'State size', state_size
	print 'Action size', action_size
	# policy_net = policy_network(state_size, action_size)
	policy_net = load_model('models/policy_11000_8.00')

	episode_reward_tracker = []
	episode_len_tracker = []
	nb_episodes = 200000
	max_episode_steps = 200
	model_save_freq = 500
	update_freq = 10
	training = True
	queue = []
	gema = 0.99

	# real_q = json.loads(open('Taxi_q').readline())
	# print real_q, len(real_q)
	for t in xrange(nb_episodes):
		observation = env.reset()
		whole_reward = 0
		states = []
		actions = []
		rewards = []
		for step in xrange(max_episode_steps):
			state = one_hot_encoder(observation, state_size)
			action = select_action(policy_net, state)
			next_observation, reward, done, _ = env.step(action)
			whole_reward += reward
			if training:
				states.append(state)
				actions.append(action)
				rewards.append(reward)

			if done:
				# Save model
				episode_reward_tracker.append(whole_reward)
				episode_len_tracker.append(step+1)
				ave_reward = float(sum(episode_reward_tracker[-100:])) / 100
				ave_len = float(sum(episode_len_tracker[-100:])) / 100
				print t, step, ave_reward, ave_len, whole_reward
				if training:
					rewards = decay_rewards(rewards, gema)
					queue.append((states, actions, rewards))  
				if training and t % update_freq == 0:
					train(policy_net, queue, action_size)
					queue = []
				if training and (t % model_save_freq == 0):
					plot(episode_reward_tracker, episode_len_tracker)
					policy_net.save('models/policy_%d_%.2lf' % (t, ave_reward))
				break
				
			else:
				observation = next_observation

if __name__ == '__main__':
	run()