from architectures import policy_network
from util import one_hot_encoder, plot, p_hot_encoder
import numpy as np
import gym 


def select_action(policy_net, state):
	probs = policy_net.predict(np.asarray([state]))[0]
	print probs
	action = np.random.choice(len(probs), p=probs)
	return action 


def train(policy_net, queue, action_size):
	gema = 0.99 
	
	total_reward = 0
	decay_rewards = []
	for frame in queue[::-1]:
		total_reward *= gema
		reward = frame[2]
		total_reward += reward
		decay_rewards.append(total_reward)
	# batch normalization
	decay_rewards = (decay_rewards - np.mean(decay_rewards)) / np.std(decay_rewards)
	
	x = []
	y = []
	for decay_r, frame in zip(decay_rewards, queue[::-1]):
		state, action, _ = frame
		x.append(state)
		label = p_hot_encoder(action, action_size, decay_r)
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
	policy_net = policy_network(state_size, action_size)

	episode_reward_tracker = []
	episode_len_tracker = []
	nb_episodes = 200000
	max_episode_steps = 200
	model_save_freq = 500
	training = True

	for t in xrange(nb_episodes):
		observation = env.reset()
		queue = []
		whole_reward = 0
		for step in xrange(max_episode_steps):
			state = one_hot_encoder(observation, state_size)
			action = select_action(policy_net, state)
			next_observation, reward, done, _ = env.step(action)
			whole_reward += reward
			if training:
				queue.append((state, action, reward))

			if done:
				# Save model
				episode_reward_tracker.append(whole_reward)
				episode_len_tracker.append(step+1)
				ave_reward = sum(episode_reward_tracker[-100:]) / 100
				ave_len = sum(episode_len_tracker[-100:]) / 100
				print t, step, ave_reward, ave_len, whole_reward 
				if training:
					train(policy_net, queue, action_size)
				if t % model_save_freq == 0:
					plot(episode_reward_tracker, episode_len_tracker)
					policy_net.save('models/policy_%d_%.2lf' % (t, ave_reward))
				break
				
			else:
				observation = next_observation

if __name__ == '__main__':
	run()