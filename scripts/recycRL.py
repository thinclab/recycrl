#!/usr/bin/env python3

"""
This script provides the Actor() and RecycRL() Classes for specifying and training a policy
using the trained reward model
"""

import os
import copy
import torch
import numpy as np
import torch.nn as nn
from csv import writer
from reward_model import RewardModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action):
        # Initialize object with everything from parent class
        super(Actor, self).__init__()

        # Convert lists to tensors and register as buffers so torch.save() saves these values
        self.register_buffer("min_action", torch.tensor(min_action))
        self.register_buffer("max_action", torch.tensor(max_action))
        print(self.min_action)

        # Define the structure of the actor network
        # state dimension input -> 512 features, 512 features -> 512 features, 512 features -> action dimension output
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        # Pass input through network and restrict between min and max action values
        return self.min_action + (torch.tanh(self.actor(x)) + 1.0) * 0.5 * (self.max_action - self.min_action)


class RecycRL(object):
    def __init__(
        self,
        state_dim=4,
        action_dim=6,
        min_action=[-0.10, -0.10, -0.01, -0.7853981634, -0.7853981634, -0.7853981634],
        max_action=[0.10, 0.10, 0.10, 0.7853981634, 0.7853981634, 0.7853981634],
        expl_noise=[0.02, 0.02, 0.02, 0.10, 0.10, 0.10],
        noise_clip=[0.04, 0.04, 0.04, 0.20, 0.20, 0.20],
        model_path="~/RecycRL/recycrl",
    ):
        # Define NN for actor and copy it for target network, then define optimizer
        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # Initialize the RewardModel Class
        self.reward_model = RewardModel(state_dim, action_dim)

        # Expand the user to handle "~"
        model_path = os.path.expanduser(model_path)

        # Define global variables
        self.action_dim = action_dim  # Dimension size of action
        self.min_action = self.actor.min_action  # Array of corresponding minimum continuous action values
        self.max_action = self.actor.max_action  # Array of corresponding maximum continuous action value
        self.expl_noise = torch.tensor(expl_noise, device=device)  # Amount of noise to add to action
        self.noise_clip = torch.tensor(noise_clip, device=device)  # Maximum noise to add
        self.total_it = 0  # Tracking variable for number of iterations
        self.prev_rewards = []  # Tracking variable for previous rewards

        # If the reward model is found
        if os.path.exists(f"{model_path}_reward_model"):
            # Load the reward model
            self.reward_model.load(model_path)

        # If the reward model cannot be found
        elif not os.path.exists(f"{model_path}_reward_model"):
            return

    def select_action(self, state, add_noise=False):
        # Convert the state to a row vector tensor and then add it to the GPU
        state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)

        # Pass the state through the actor network to get the action
        action = self.actor(state)

        # If the "add_noise" flag is set to True
        if add_noise:
            # Get random noise based on Gaussian with 0 mean and "expl_noise" variance with action size
            noise = (torch.randn_like(action) * self.expl_noise).clamp(-self.noise_clip, self.noise_clip)

            # Add the noise to the action
            action = action + noise

        # Clamp final action to valid bounds
        action = torch.max(torch.min(action, self.max_action), self.min_action)

        return action.cpu().detach().numpy().flatten()

    def train(self, batch):
        # Increment the iteration tracking variable
        self.total_it += 1

        # Extract the corresponding values from the passed batch
        state, _, _ = batch

        # Pass state and actor's action through the reward model to get the mean and variance of the expected reward
        mean, std = self.reward_model.reward_model.predict(state, self.actor(state))

        loss = -(mean - 0.4 * std).mean()

        # Print the loss every 1000 iterations
        if self.total_it % 1000 == 0:
            print("Mean:", mean.mean().item())
            print("Std:", std.mean().item())
            print("Actor action mean:", self.actor(state).mean().item())
            print(f"Actor Loss: {loss.item()}")

        # Optimize the actor
        self.actor_optimizer.zero_grad()  # Clear old gradient
        loss.backward()  # Backpropagate the loss
        self.actor_optimizer.step()  # Update the parameters

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        np.save(filename + "_total_it.npy", self.total_it)
        np.save(filename + "_prev_rewards.npy", self.prev_rewards)

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.total_it = int(np.load(filename + "_total_it.npy"))
        self.prev_rewards = np.load(filename + "_prev_rewards.npy").tolist()

        self.actor.min_action.copy_(torch.tensor(self.min_action, device=device))
        self.actor.max_action.copy_(torch.tensor(self.max_action, device=device))

    def evaluate_policy(self, reward, filename):
        # Add the most recent reward to the list of previous rewards
        self.prev_rewards.append(reward)

        # If there are more than 20 previous rewards
        if len(self.prev_rewards) >= 20:
            # Calculate the average reward over the last 20 rewards
            avg_reward = sum(self.prev_rewards) / 20

            # Append the average reward and current iteration to a CSV file
            with open(filename + "_average_rewards.csv", mode="a") as file:
                write_file = writer(file)
                write_file.writerow([f"{self.total_it}: {avg_reward}"])

            # Remove the oldest reward
            self.prev_rewards.pop(0)

        # Otherwise, the reward average is 0
        else:
            avg_reward = 0

        return avg_reward
