import numpy as np
import gym
import time
import json
from collections import defaultdict


def explore():
	env = gym.make("Taxi-v2")
	transitions = []
	destination = []
	episodes = 600
	max_episode_step = 200
	for t in xrange(episodes):
		observation = env.reset()
		for step in xrange(max_episode_step):
			action = env.action_space.sample()
			next_observation, reward, done, info = env.step(action)
			transitions.append((observation, action, next_observation, reward))
			if done:
				if step != max_episode_step-1:
					destination.append(observation)
				break
			else:
				observation = next_observation

	env.close()
	return (transitions, list(set(destination)))


def planning(transitions):
	# Build graph
	dist = defaultdict(dict)
	all_nodes = []

	trans_table = {}
	for i in transitions:
		all_nodes.append(i[0])
		all_nodes.append(i[2])
		trans_table[(i[0], i[1])] = (i[2], i[3])
		dist[i[0]][i[2]] = i[3]

	all_nodes = set(all_nodes)
	for i in all_nodes:
		for j in all_nodes:
			dist[i][j] = -1e9
			if i == j:
				dist[i][j] = 0

	for i in transitions:
		x = i[0]
		y = i[2]
		if x == y:
			continue
		dist[x][y] = max(dist[x][y], i[3])

	for i in all_nodes:
		for j in all_nodes:
			dist[i][j] = -dist[i][j]

	# floyd algorithm
	for k in all_nodes:
		for i in all_nodes:
			for j in all_nodes:
				dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])

	for i in dist:
		for j in dist[i]:
			dist[i][j] = -dist[i][j]

	return (trans_table, dist)


def cal_q(trans_table, dist, destination):
	Q = {}
	for i in dist:
		for action in range(6):
			next_state, reward = trans_table[(i, action)]
			q_val = 20 + (reward if reward < 0 else 0) + max(dist[next_state][des] for des in destination)
			Q[str((i, action))] = q_val
	return Q 


def play(Q):
	from gym import wrappers
	env = gym.make("Taxi-v2")
	# env = wrappers.Monitor(env, '/tmp/taxi-model_based', force=True)
	episodes = 10
	max_episode_step = 200
	for t in xrange(episodes):
		observation = env.reset()
		env.render()
		for step in xrange(max_episode_step):	
			time.sleep(1)
			q_vals = [Q[str((observation, i))] for i in range(6)]
			action = np.argmax(q_vals)
			# print 
			# print q_vals
			# print 
			next_observation, reward, done, info = env.step(action)
			env.render()
			if done:
				break
			else:
				observation = next_observation
	# env.close()
	# gym.upload('/tmp/taxi-model_based', api_key='sk_mcj9MFqJRTSuLMNoP7eEZQ')


transitions, destination = explore()
trans_table, dist = planning(transitions)
print len(dist), len(trans_table)
Q = cal_q(trans_table, dist, destination)
play(Q)
with open('Taxi_q', 'w') as f:
	tmp = json.dumps(Q)
	f.write(tmp)
# for i in dist:
# 	for j in dist[i]:
# 		if dist[i][j] >= 20:
# 			print i, j, dist[i][j]