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
        model_path="~/RecycRL",
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

        # If the reward model has been saved previously, load the model
        if os.path.exists(f"{model_path}/Reward_Model"):
            self.reward_model.load(model_path)

    def select_action(self, state, add_noise=False):
        # Convert the state to a row vector tensor and then add it to the GPU
        state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)

        # Pass the state through the actor network to get the action
        action = self.actor(state)

        # print(action.cpu().detach().numpy().flatten())

        # If the "add_noise" flag is set to True
        if add_noise:
            # Get random noise based on Gaussian with 0 mean and "expl_noise" variance with action size
            noise = (torch.randn_like(action) * self.expl_noise).clamp(-self.noise_clip, self.noise_clip)

            # print(noise.cpu().detach().numpy().flatten())

            # Add the noise to the action
            action = action + noise

        # Clamp final action to valid bounds
        action = action.clamp(self.min_action, self.max_action)
        # action = torch.max(torch.min(action, self.max_action), self.min_action)

        return action.cpu().detach().numpy().flatten()

    def train(
        self,
        batch,
        beta=0.6,
        base_threshold=0.95,
        perturbation_threshold=0.80,
        step=[0.001, 0.001, 0.001, 0.005, 0.005, 0.005],
        max_delta=[0.05, 0.05, 0.05, 0.25, 0.25, 0.25],
        use_baseline=False,
        print_iterations=100,
    ):
        # Increment the iteration tracking variable
        self.total_it += 1

        # Extract the states from the passed batch
        states, _, _ = batch

        # Get the actor's/policy's actions for each state
        actions = self.actor(states)

        # Pass states and actor's actions through the reward model to get the mean and variance of the expected reward
        means, stds = self.reward_model.reward_model.predict(states, actions)

        # Get the scores of the actions which is the lower confidence bound
        scores = means - beta * stds

        # Get the perturbation score for the base actions
        perturbation_scores = self.get_perturbation_score(states, actions, perturbation_threshold, step, max_delta)

        # Generate a boolean mask on the actions which represents actions with scores that are above the threshold
        passed_mask = scores >= base_threshold

        # Convert the boolean mask to floats in order to multiply later (False -> 0.0 and True -> 1.0)
        passed_scores = passed_mask.float()

        # Multiply mask of floats by perturbation scores to keep scores for base actions that are above the threshold
        passed_perturbation_scores = passed_scores * perturbation_scores

        # If the user wants to train with a baseline
        if use_baseline:
            # Calculate the combined score for each action
            combined_scores = scores + passed_perturbation_scores

            # Calculate the batch's average combined score for a baseline
            baseline = combined_scores.mean()

            # Calculate the loss with the combined scores and baseline
            loss = -(combined_scores - baseline).mean()

        # If the user does not want to train with a baseline
        elif not use_baseline:
            # Calculate the loss which is the expected rewards plus the perturbation scores of passed actions
            loss = -(scores + passed_perturbation_scores).mean()

        # Print the loss after we have reached a multiple of 'print_iterations'
        if self.total_it % print_iterations == 0:
            print("Perturbation Score", passed_perturbation_scores.mean())
            print("Mean:", means.mean().item())
            print("Std:", stds.mean().item())
            print(f"Actor Loss: {loss.item()}")
            print("")

        # Optimize the actor
        self.actor_optimizer.zero_grad()  # Clear old gradient
        loss.backward()  # Backpropagate the loss
        self.actor_optimizer.step()  # Update the parameters

    def get_perturbation_score(
        self,
        states,
        actions,
        pert_thresh,
        step=[0.001, 0.001, 0.001, 0.005, 0.005, 0.005],
        max_delta=[0.05, 0.05, 0.05, 0.25, 0.25, 0.25],
    ):
        # Get the batch size and action dimension size from the passed actions
        batch_size, action_dim = actions.shape

        # Convert the "step" and "max_delta" lists to tensors and place on device
        step = torch.tensor(step, device=device)
        max_delta = torch.tensor(max_delta, device=device)

        # Define tensors of all zeros of size (batch_size, action_dim) to hold current noise
        # increment per dimension in negative and positive directions to be applied to actions
        delta_pos = torch.zeros(batch_size, action_dim, device=device)
        delta_neg = torch.zeros(batch_size, action_dim, device=device)

        # Define entirely True boolean tensors of size (batch_size, action_dim) which represent
        # which action dimensions can still be incremented in both the negative and positive directions
        active_pos = torch.ones(batch_size, action_dim, dtype=torch.bool, device=device)
        active_neg = torch.ones(batch_size, action_dim, dtype=torch.bool, device=device)

        # Define an identity matrix of size (action_dim, action_dim)
        identity = torch.eye(action_dim, device=device)

        # Loop while there are dimensions that can have more incremental noise added
        while active_pos.any() or active_neg.any():
            # If any action dimensions can have positive noise added
            if active_pos.any():
                # Generate a tensor of the base actions with perturbations in each of the action dimensions
                # Convert the "actions" tensor from (batch_size, action_dim) -> (batch_size, 1, action_dim)
                # Convert the "delta_pos" tensor from (batch_size, action_dim) -> (batch_size, action_dim, 1)
                # Multiply "delta_pos" by "identity" to get per dimension noise for each action (b, a, 1) * (a, a) -> (b, a, a)
                # Add per dimension noise to all actions (b, a, 1) + (b, a, a) -> (b, a, a)
                perturbed_actions = actions.unsqueeze(1) + delta_pos.unsqueeze(2) * identity

                # Convert "perturbed_actions" from (b, a, a) -> (b * a, a) to pass through reward model
                perturbed_actions_flat = perturbed_actions.view(batch_size * action_dim, action_dim)

                # Create a tensor of the passed states duplicated for the amount of perturbed actions
                # Convert "states" from (batch_size, state_dim) -> (batch_size, 1, state_dim)
                # Expand "states" or duplicate entries to generate array of size (batch_size, action_dim, state_dim)
                # Convert the tensor from (batch_size, action_dim, state_dim) -> (batch_size * action_dim, state_dim)
                states_flat = states.unsqueeze(1).expand(-1, action_dim, -1).reshape(batch_size * action_dim, -1)

                # Pass all of the perturbed actions with the corresponding states through the reward model
                means, stds = self.reward_model.reward_model.predict(states_flat, perturbed_actions_flat)

                # Calculate the score (LCB) for all of the perturbed actions and convert from (b * a, 1) -> (b, a)
                score = (means - 0.4 * stds).view(batch_size, action_dim)

                # Create a boolean mask for scores that are above the threshold and whose increments are less than the max
                increment_mask = (score >= pert_thresh) & (delta_pos < max_delta) & active_pos

                # Convert mask from boolean to float and multiply by step to increase the noise for passed perturbed actions
                # Then add to the current noise increment tensor to generate the new noise tensor
                delta_pos = delta_pos + increment_mask.float() * step

                # Assign the increment mask to the active_pos tensor for logic purposes
                active_pos = increment_mask

            # If any action dimensions can have negative noise added
            if active_neg.any():
                # See comments above; this section is for negative noise actions
                perturbed_actions = actions.unsqueeze(1) - delta_neg.unsqueeze(2) * identity
                perturbed_actions_flat = perturbed_actions.view(batch_size * action_dim, action_dim)
                states_flat = states.unsqueeze(1).expand(-1, action_dim, -1).reshape(batch_size * action_dim, -1)
                means, stds = self.reward_model.reward_model.predict(states_flat, perturbed_actions_flat)
                scores = (means - 0.4 * stds).view(batch_size, action_dim)
                decrement_mask = (scores >= pert_thresh) & (delta_neg < max_delta) & active_neg
                delta_neg = delta_neg + decrement_mask.float() * step
                active_neg = decrement_mask

        # After we have finished incrementing noise for all actions and the action dimensions and exited the loop, calculate the
        # maximum positive and negative perturbations by subtracting step to offset last increment and clamping to "max_delta"
        max_pos = torch.clamp(delta_pos - step, max=max_delta)
        max_neg = torch.clamp(delta_neg - step, max=max_delta)

        # Calculate total perturbation score for actions by adding per dimension positive and negative max perturbation
        total_perturbation_scores = (max_pos + max_neg).sum(dim=1)

        # Normalize the perturbations score by dividing by the "max_delta" sum multiplied by two (for negative and positive)
        perturbation_scores = total_perturbation_scores / 2 * max_delta.sum()

        return perturbation_scores

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "/Policy")
        torch.save(self.actor_optimizer.state_dict(), filename + "/Policy_Optimizer")
        np.save(filename + "/Policy_Iterations.npy", self.total_it)
        np.save(filename + "/Policy_Previous_Rewards.npy", self.prev_rewards)

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "/Policy"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "/Policy_Optimizer"))
        self.total_it = int(np.load(filename + "/Policy_Iterations.npy"))
        self.prev_rewards = np.load(filename + "/Policy_Previous_Rewards.npy").tolist()

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
