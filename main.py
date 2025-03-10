# import gym
 
# env=gym.make("FrozenLake-v1", render_mode="human")
# env.reset()
# #generate random action
# randomAction= env.action_space.sample()
# returnValue = env.step(1)
# # actions: left -0, down - 1, right - 2, up- 3
# env.action_space
# # format of returnValue is (observation,reward, terminated, truncated, info)
# # observation (object)  - observed state
# # reward (float)        - reward that is the result of taking the action
# # terminated (bool)     - is it a terminal state
# # truncated (bool)      - it is not important in our case
# # info (dictionary)     - in our case transition probability
# print(returnValue)
 
# env.render()


# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 21:29:24 2022
 
@author: ahaber
"""
import gym
import time
import numpy as np
env=gym.make("FrozenLake-v1",render_mode='human')
env.reset()
env.render()
# select the discount factor
discountFactor=0.9
# initialize the value function vector
valueFunctionVector=np.zeros(env.observation_space.n)
maxNumberOfIterations=1000
convergenceTolerance=10**(-6)
 
convergenceTrack=[]
 
for iterations in range(maxNumberOfIterations):
    convergenceTrack.append(np.linalg.norm(valueFunctionVector,2))
    valueFunctionVectorNextIteration=np.zeros(env.observation_space.n)
    for state in env.P:
        outerSum=0
        for action in env.P[state]:
            innerSum=0
            for probability, nextState, reward, isTerminalState in env.P[state][action]:
                #print(probability, nextState, reward, isTerminalState)
                innerSum=innerSum+ probability*(reward+discountFactor*valueFunctionVector[nextState])
                print(f"state {state} action {action} reward {reward}")
            outerSum=outerSum+0.25*innerSum
        valueFunctionVectorNextIteration[state]=outerSum
    if(np.max(np.abs(valueFunctionVectorNextIteration-valueFunctionVector))<convergenceTolerance):
        valueFunctionVector=valueFunctionVectorNextIteration
        print('Converged!')
        break
    valueFunctionVector=valueFunctionVectorNextIteration          