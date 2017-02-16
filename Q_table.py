'''
Simple Q-learning example
Q-learning equation:
	Q(s,a) = r + γ(max(Q(s’,a’))
'''

import gym
import numpy as np

env = gym.make('Taxi-v2')

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .85
γ = .99
num_episodes = 2000
total_reward = 0
for i in range(num_episodes):
    # Reset environment; in starting state
    state = env.reset()
    for j in range(99):
        # Select action from Q-table given current state
        a = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        # Pass new action to enviornment
        observation,reward,done,_ = env.step(a)
        # Q function: Q(s,a) = r + γ(max(Q(s’,a’))
        Q[state,a] = Q[state,a] + lr*(reward + γ*np.max(Q[observation,:]) - Q[state,a])
        total_reward += reward
        state = observation
        if done == True:
            break

print("Score over time: " +  str(total_reward/num_episodes))
print("Final Q-Table Values")
print(Q)