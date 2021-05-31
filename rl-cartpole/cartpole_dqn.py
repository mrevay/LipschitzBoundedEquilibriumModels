# Learn Deep Q Networks for the standard cart-pole benchmarking 
# problem defined in Open AI gym.
#
# Skeleton code by: Tejaswi S. Digumarti
# Edits and LBEN adaptation: Nicholas H. Barbara
# Email: nbar5346@uni.sydney.edu.au

import gym
import json
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional
import torch.optim as optim

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import NODEN
from splitting import MONPeacemanRachford
from train import MON_DEFAULTS, expand_args, run_tune_alpha


class FCNetwork(nn.Module):
    """
    Fully-connected neural network with 3 layers, 
    ReLu activation function.
    """

    def __init__(self, input_size, output_size):
        """Define the layers of the network."""

        # Network dimensions
        self.input_size = input_size        # 4 states
        self.output_size = output_size      # 2 actions
        self.hidden = 256                   # (256)

        super(FCNetwork, self).__init__()

        # Define the layers of the network
        self.fc1 = nn.Linear(self.input_size, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.output_size)

    def forward(self, x):
        """Define the forward pass."""

        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# TODO: Need to ensure operator splitting converges
class NODEN_Lip_Net(nn.Module):
    """Modified from train.py"""

    def __init__(self, input_size, output_size, gamma, **kwargs):
        """Define the network structure."""

        # Network dimensions
        self.input_size = input_size
        self.output_size = output_size
        self.width = 256
        self.m = 0.1

        super().__init__()

        # Define network structure
        linear_module = NODEN.NODEN_Lipschitz_Fc(self.input_size, self.width, self.output_size, gamma, m=self.m)
        nonlin_module = NODEN.NODEN_ReLU()
        self.mon = MONPeacemanRachford(linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))

    def forward(self, x):
        """Define the forward pass."""

        x = x.view(x.shape[0], -1)
        z = self.mon(x)
        y = self.mon.linear_module.G(z[-1])

        return y


class DQNCartPoleLearner():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Q-learning hyperparameters
        self.discount_factor = 0.99     # discount factor on future rewards (0.99)
        self.learning_rate   = 1e-4     # learning rate for the network (1e-4)
        self.batch_size      = 128      # batch size (128)
        self.train_start     = 1000     # number of samples to add to memory before training (1000)

        # Parameters of exploration vs. exploitation
        self.epsilon_max     = 0.95     # probability of exploration, initially only explore (0.95)
        self.epsilon_decay   = 0.9998   # amount by which to decay after every action (v0: 0.999)/(v1: 0.9998)
        self.epsilon_min     = 0.005    # minimum value of epsilon (0.005)
        self.epsilon         = self.epsilon_max

        # Memory to store the tuples of (state, action, reward, next_state, done) (14e3)
        self.memory_size = 14000
        self.memory = deque(maxlen=self.memory_size)

        # Gradient clipping (improves algorithmic stability)
        self.clipping = 1.0

        # Create the model
        self.gamma = None
        # self.gamma = 240                          # Lipschitz bound for LBEN
        self.alpha = 0.5                            # Basically the step-size for operator splitting (eg: Peaceman-Rachford)
        self.model = self.create_nn_model()         # Train this model
        self.old_model = self.create_nn_model()     # Use this model to get the expected Q_value
        self.update_old_model()

        # Define the optmizer for the network
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # For plotting/saving (save the loss later!!)
        self.name = "cartpole_lben_02"
        self.loss = []

    def make_hyperparams_dict(self):
        """Store important hyperparameters/information in a dictionary"""
        hyper_dict = {
            "name":             self.name,
            "discount_fact":    self.discount_factor,
            "learning_rate":    self.learning_rate,
            "batch_size":       self.batch_size,
            "train_start":      self.train_start,
            "epsilon_set":      (self.epsilon_max, self.epsilon_decay, self.epsilon_min),
            "memory_size":      self.memory_size,
            "clipping":         self.clipping,
            "optimizer":        "Adam",
            "lossfunc":         "MSE",
            "lben_gamma":       self.gamma,
            "lben_alpha":       self.alpha
        }
        return hyper_dict

    def create_nn_model(self):
        """
        Create the Neural Network model that approximates the Q-value function
        :return: The network that approximates the Q-value function
        :rtype: FCNetwork
        """

        # TODO: If need be, try tuning alpha for operator splitting
        if self.gamma is None:
            return FCNetwork(self.state_size, self.action_size)
        return  NODEN_Lip_Net(self.state_size, self.action_size, self.gamma, 
                              alpha=self.alpha, verbose=False)

    def choose_action(self, state):
        """
        Choose an action based on an epsilon greedy exploration vs. exploitation method, 
        i.e. choose a random action (explore) with probability (epsilon) or
        the action with max q-value (exploit) with probability (1-epsilon)

        :param state: the current state in which the agent is
        :type state: array
        :return: the action to perform
        :rtype: int
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.from_numpy(state)
            q_values = self.model(state_tensor).detach().numpy()[0]
            return np.argmax(q_values)

    def add_sample_to_memory(self, state, action, reward, next_state, done):
        """
        Adds a tuple of (state, action, reward, next_state, done) to the memory
        :param state: the current state
        :type state:
        :param action: the action taken
        :type action: int
        :param reward: the reward attained
        :type reward: float
        :param next_state: next state
        :type next_state:
        :param done: has the episode finished or not
        :type done: bool
        """
        self.memory.append((state, action, reward, next_state, done))
        if (self.epsilon > self.epsilon_min) and (len(self.memory) > self.train_start):
            self.epsilon *= self.epsilon_decay

    def update_old_model(self):
        model_weights = self.model.state_dict()
        self.old_model.load_state_dict(model_weights)

    def vector_to_tensor(self, vector):
        return torch.from_numpy(vector.astype(np.float32))

    def train_model(self):
        
        # No training at the start. Add random samples to memory
        if len(self.memory) < self.train_start:
            return

        # Sample batch_size samples from memory
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        input_states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            input_states[i] = mini_batch[i][0]      # state
            action.append(mini_batch[i][1])         # action
            reward.append(mini_batch[i][2])         # reward
            next_states[i] = mini_batch[i][3]       # next state
            done.append(mini_batch[i][4])           # done

        # Fill the predicted Q values from the current state using the current model
        input_states_tensor = self.vector_to_tensor(input_states)
        q_predicted = self.model(input_states_tensor)

        # Initialise expected q-values
        q_expected = self.model(input_states_tensor)
        
        # Fill the Q values from the next state using the previous model
        # Method similar to: https://www.nature.com/articles/nature14236
        next_states_tensor = self.vector_to_tensor(next_states)
        q_next_state = self.old_model(next_states_tensor)

        # Update the predicted Q values only of the chosen action
        for i in range(self.batch_size):
            if done[i]:
                q_expected[i][action[i]] = reward[i]
            else:
                q_expected[i][action[i]] = reward[i] + \
                    self.discount_factor * (np.max(q_next_state.detach().numpy()[i]))

        # Get loss, compute gradients, and update the model parameters
        # Add gradient clipping for numerical stability
        self.loss = self.criterion(q_expected, q_predicted)
        self.loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping)
        self.optimizer.step()


if __name__ == '__main__':

    # Useful parameters
    num_episodes  = 2000
    render        = False
    great_success = False

    # Create the cart pole environment from the open AI gym
    env = gym.make('CartPole-v1')
    max_score = env.spec.max_episode_steps
    threshold = env.spec.reward_threshold

    # Print some information about the environment
    state_size = env.observation_space.shape[0]
    print("Size of the state = {}".format(state_size))
    action_size = env.action_space.n
    print("Size of the action space = {}".format(action_size))

    # Initialise the agent
    agent = DQNCartPoleLearner(state_size, action_size)

    # For plotting/saving
    scores = []
    mean_scores = []
    episodes = []
    plt.figure()
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show(block=False)

    for e in range(num_episodes):
        done = False
        score = 0
        mean_score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size]).astype(np.float32)

        while not done:

            if render:
                env.render()

            # Perform an action and go to next step
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size]).astype(np.float32)

           # Keep reward if big score or still going, 
           # otherwise we failed (big negative reward)
            if not done or score == (max_score-1):
                reward = reward
            else:
                reward = -100

            # Add to memory, train the model, and update score/state
            agent.add_sample_to_memory(state, action, reward, next_state, done)
            agent.train_model()
            score += reward
            state = next_state

            # When the game (episode) is over...
            if done:

                # Update old model params at the end of each EPISODE 
                # See Mnih et al. 2015 for different approach
                agent.update_old_model()

                # Show how we performed in the episode
                if score == max_score:
                    score = score
                else:
                    score += 100
                scores.append(score)

                # Compute mean score over the past 100 episodes
                if e > 100:
                    mean_score = np.mean(scores[e-100:e])
                    mean_scores.append(mean_score)
                else:
                    mean_scores.append(np.mean(scores))

                # Stop if we meet the goal
                if (mean_scores[e] >= threshold) and (not great_success):
                    great_success = True
                    break

                # Print values
                print("Episode: {}, score: {}, mean score: {:.2f}, epsilon: {:.3f}".format(
                    e, scores[e], mean_scores[e], agent.epsilon))

                # Plot values
                episodes.append(e)
                if e > 0:
                    if len(agent.memory) > agent.train_start:
                        plt.plot(episodes[(e-1):(e+1)], scores[(e-1):(e+1)], 'b')
                    else:
                        plt.plot(episodes[(e-1):(e+1)], scores[(e-1):(e+1)], 'g')
                    plt.plot(episodes[(e-1):(e+1)], mean_scores[(e-1):(e+1)], 'r')
                    plt.pause(0.0001)
        
        # Stop if we achieved the reward threshold
        if great_success:
            break
    
    # Save the network model
    if great_success:
        fpath = "saved_models/" + agent.name
    else:
        fpath = "saved_models/nogood/" + agent.name
    torch.save(agent.model.state_dict(), fpath + ".pt")

    # Save the hyperparameters, rewards, and random seed
    save_dict = agent.make_hyperparams_dict()
    save_dict["scores"] = scores
    save_dict["mean_scores"] = mean_scores
    with open(fpath + ".json", "w") as f:
        json.dump(save_dict, f)
    plt.show()


def estimate_gamma(agent):
    """
    Estimate the Lipschitz constant of an FCNetwork object.
    This is not great code, hacking things together last minute.
    """

    weights = []
    weights.append(agent.model.state_dict()["fc1.weight"].numpy())
    weights.append(agent.model.state_dict()["fc2.weight"].numpy())
    weights.append(agent.model.state_dict()["fc3.weight"].numpy())

    lip_est = 1.0
    for i in range(3):
        _, s, _ = np.linalg.svd(weights[i])
        lip_est *= s.max()

    return lip_est