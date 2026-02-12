#!/usr/bin/env python3

"""
This script provides the Model() and RewardModel() Classes for specifying and training the reward model from data
"""

import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, state_dim=4, action_dim=6, network_amount=5):
        super(Model, self).__init__()

        # Define the structure for each of the models
        self.models = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim + action_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                )
                for _ in range(network_amount)
            ]
        )

    def forward(self, state, action):
        # Concatenate the state and action inputs into a single tensor
        sa = torch.cat([state, action], 1)

        # Pass the converted input through the networks
        return [model(sa) for model in self.models]

    def predict(self, state, action):
        # Pass the state and action through the networks
        raw_outputs = self.forward(state, action)

        # Apply sigmoid to each of the raw outputs and stack the converted outputs
        probabilities = torch.stack([torch.sigmoid(output) for output in raw_outputs], dim=0)

        # Calculate the average and variance of the network probabilities
        mean = probabilities.mean(dim=0)
        std = probabilities.std(dim=0)

        return mean, std


class RewardModel(object):
    def __init__(self, state_dim=4, action_dim=6, network_amount=5):
        # Define NN for reward model and the optimizer
        self.reward_model = Model(state_dim, action_dim, network_amount).to(device)
        self.reward_model_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=3e-4)
        self.criterion = nn.BCEWithLogitsLoss()

        # Define global variables
        self.action_dim = action_dim
        self.total_it = 0  # Tracking variable for number of iterations
        self.prev_rewards = []  # Tracking variable for previous rewards

    def train(self, batch):
        # Increment the iteration tracking variable
        self.total_it += 1

        # Extract the corresponding values from the passed batch
        state, action, reward = batch

        # Get the raw predicted rewards from the reward models
        predicted_logits = self.reward_model(state, action)

        # Manipulate reward shape to be used for BCELoss
        reward = reward.view(-1, 1).float()

        # Define loss and batch size variables for upcoming loop
        loss = 0
        batch_size = state.size(0)

        # Loop through each networks' predicted reward
        for logit in predicted_logits:
            # Generate random indices
            index = torch.randint(0, batch_size, (batch_size,), device=device)

            # Generate different batch based on random indices
            reward_batch = reward[index]
            logit_batch = logit[index]

            # Increment the loss with the binary cross-entropy loss of the current network prediction
            loss += self.criterion(logit_batch, reward_batch)

        # Calculate the average loss
        loss = loss / len(predicted_logits)

        # Print the loss every 1000 iterations
        if self.total_it % 1000 == 0:
            print(f"Reward Model Loss: {loss.item()}")

        # Optimize the critics
        self.reward_model_optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Backpropagate the loss
        self.reward_model_optimizer.step()  # Update the parameters

    def save(self, filename):
        torch.save(self.reward_model.state_dict(), filename + "_reward_model")
        torch.save(self.reward_model_optimizer.state_dict(), filename + "_reward_model_optimizer")
        np.save(filename + "reward_model_total_it.npy", self.total_it)

    def load(self, filename):
        self.reward_model.load_state_dict(torch.load(filename + "_reward_model"))
        self.reward_model_optimizer.load_state_dict(torch.load(filename + "_reward_model_optimizer"))
        self.total_it = int(np.load(filename + "_reward_model_total_it.npy"))
