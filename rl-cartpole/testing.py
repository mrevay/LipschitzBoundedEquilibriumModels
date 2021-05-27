import gym
import json
import torch
import cvxpy
import numpy as np
import matplotlib.pyplot as plt

from cartpole_dqn import FCNetwork, NODEN_Lip_Net, estimate_gamma


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


# Read in a model and its info
# TODO: Need to make sure the hyperparameters are the same as initialisation?
fname = "saved_models/cartpole_lben_00"
if "lben" in fname:
    with open(fname + ".json") as f:
        hyper_dict = json.load(f)
    model = NODEN_Lip_Net(4,2, gamma=hyper_dict["lben_alpha"], verbose=False)
else:
    model = FCNetwork(4,2)
model.load_state_dict(torch.load(fname + ".pt"))
agent = LearnedAgent(model)

# Try to test it out on the cart-pole environment
env = gym.make('CartPole-v0')
state = env.reset()
my_error = 0.0
for _ in range(env.spec.max_episode_steps):
    env.render()
    a = agent(state)
    out = env.step(a)
    state = out[0] + my_error*np.random.randn()
    print(_)
env.close()