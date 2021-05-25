import gym
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# import lben
# import NODEN <----- NEED CVXPY!!
# import mon

from cartpole_dqn import FCNetwork


class LearnedAgent():
    """To help with evaluating a model"""

    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def __call__(self, state):
        """
        Choose an action based on max q-value

        :param state: the current environment state
        :type state: array
        :return: the action to perform
        :rtype: int
        """
        state = np.reshape(state, [1, 4]).astype(np.float32)
        state_tensor = torch.from_numpy(state)
        q_values = self.model(state_tensor).detach().numpy()[0]
        return np.argmax(q_values)

# Read in one of the models


# Read in a model and its info
fname = "saved_models/cartpole_03"
model = FCNetwork(4,2)
model.load_state_dict(torch.load(fname + ".pt"))
agent = LearnedAgent(model)

with open(fname + ".json") as f:
  hyper_dict = json.load(f)

# Try to test it out on the cart-pole environment
env = gym.make('CartPole-v0')
state = env.reset()
my_error = 0.0
for _ in range(200):
    env.render()
    a = agent(state)
    out = env.step(a)
    state = out[0] + my_error
env.close()