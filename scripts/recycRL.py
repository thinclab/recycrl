#!/usr/bin/env python3

"""
This script provides the Actor(), RecycRL(), A2P(), NRMDP() classes for training a policy
using the trained reward model and differing objectives
"""

import os
import torch
import numpy as np
import torch.nn as nn
from reward_model import RewardModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action, initial_action=None):
        # Initialize object with everything from parent class
        super(Actor, self).__init__()

        # Convert lists to tensors and register as buffers so torch.save() saves these values
        self.register_buffer("min_action", torch.tensor(min_action))
        self.register_buffer("max_action", torch.tensor(max_action))

        # Define the structure of the actor network
        # state dimension input -> 512 features, 512 features -> 512 features, 512 features -> action dimension output
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

        # Apply Xavier/Glorot initialization
        self._init_weights()

        # If an initial action is provided
        if initial_action:
            # Convert the initial action to a tensor
            initial_action = torch.tensor(initial_action, dtype=torch.float32)

            # Normalize desired action to [-1, 1] for inverse tanh
            y = 2.0 * (initial_action - self.min_action) / (self.max_action - self.min_action) - 1.0

            # Clamp the action to avoid numerical issues at exactly -1 or 1
            y = torch.clamp(y, -0.999999, 0.999999)

            # Compute inverse tanh
            bias = torch.atanh(y)

            # Set the bias for the initial action
            self.actor[4].bias.data.copy_(bias)

    def _init_weights(self):
        # Loop through each module
        for m in self.modules():
            # If the module is a linear layer
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(m.weight, gain=1.5)

                # Small bias initialization
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Pass input through network and restrict between min and max action values
        return self.min_action + (torch.tanh(self.actor(x)) + 1.0) * 0.5 * (self.max_action - self.min_action)


# This is class for PAR objective
class RecycRL(object):
    def __init__(
        self,
        state_dim=4,
        action_dim=6,
        min_action=[-0.10, -0.10, -0.01, -0.7853981634, -0.7853981634, -0.7853981634],
        max_action=[0.10, 0.10, 0.10, 0.7853981634, 0.7853981634, 0.7853981634],
        expl_noise=[0.015, 0.015, 0.015, 0.075, 0.075, 0.075],
        noise_clip=[0.03, 0.03, 0.03, 0.15, 0.15, 0.15],
        model_path="~/RecycRL",
        initial_action=None,
    ):
        # Define NN for actor and copy it for target network, then define optimizer
        self.actor = Actor(state_dim, action_dim, min_action, max_action, initial_action).to(device)
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

        # If the reward model has been saved previously, load the model
        if model_path != "" and os.path.exists(f"{model_path}/Reward_Model"):
            self.reward_model.load(model_path)

    def select_action(self, state, add_noise=False, seed=0):
        # Convert the state to a row vector tensor and then add it to the GPU
        state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)

        # Pass the state through the actor network to get the action
        action = self.actor(state)

        # print("Action before noise", action.cpu().detach().numpy().flatten())

        # If the "add_noise" flag is set to True
        if add_noise:
            # Create a local noise generator and set the seed
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

            # Get random noise based on Gaussian with 0 mean and "expl_noise" variance with action size
            noise = torch.randn(action.shape, device=device, generator=generator)
            noise = (noise * self.expl_noise).clamp(-self.noise_clip, self.noise_clip)

            # print("Noise:", noise.cpu().detach().numpy().flatten())

            # Add the noise to the action
            action = action + noise

        # Clamp final action to valid bounds
        action = action.clamp(self.min_action, self.max_action)

        return action.cpu().detach().numpy().flatten()

    def train(
        self,
        batch,
        alpha=1.0,
        beta=2.0,
        step=[0.001, 0.001, 0.001, 0.005, 0.005, 0.005],
        max_delta=[0.03, 0.03, 0.03, 0.15, 0.15, 0.15],
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

        # If alpha is set to 0, skip integral calculation and set integral scores to 0 to save computation
        if alpha == 0.0:
            integral_scores = torch.zeros_like(scores)

        # If alpha is not 0, calculate the integral scores for the actions
        elif alpha != 0.0:
            integral_scores = self.get_integral_scores(beta, states, actions, step, max_delta)

        # Calculate the loss which is the expected rewards plus the integral scores
        loss = -((1 - alpha) * scores + alpha * integral_scores).mean()

        # Print the loss after we have reached a multiple of 'print_iterations'
        if self.total_it % print_iterations == 0:
            print("Integral Score", integral_scores.mean().item())
            print("LCB Score:", scores.mean().item())
            print(f"Actor Loss: {loss.item()}")
            print("")

        # Optimize the actor
        self.actor_optimizer.zero_grad()  # Clear old gradient
        loss.backward()  # Backpropagate the loss
        self.actor_optimizer.step()  # Update the parameters

    def train_for_buffalo(
        self,
        reward_model=None,
        online=True,
        alpha=1.0,
        beta=2.0,
        step=[0.001],
        max_delta=[0.025],
        print_iterations=100,
    ):
        # Increment the iteration tracking variable
        self.total_it += 1

        # Define state as a tensor of size 1 with value to 0 since buffalo only has one state
        state = torch.tensor([[0.0]], device=device)

        # Get the actor's/policy's action
        action = self.actor(state)

        # If we are training a policy online
        if online:
            # Assign the passed reward model to the policy's reward model
            self.reward_model.reward_model = reward_model.to(device)

            # Get the score of the action
            score = self.reward_model.reward_model(action)

        # If we are training a policy offline with a learned reward model
        elif not online:
            # Pass states and actor's actions through the reward model to get the mean and variance of the expected reward
            mean, std = self.reward_model.reward_model.predict(state, action)

            # Get the score of the actions which is the lower confidence bound
            score = mean - beta * std

        # If alpha is set to 0, skip integral calculation and set integral scores to 0 to save computation
        if alpha == 0.0:
            integral_score = torch.zeros_like(score)

        # If alpha is not 0, calculate the integral scores for the actions
        elif alpha != 0.0:
            integral_score = self.get_integral_scores(beta, state, action, step, max_delta, online)

        # Calculate the loss which is the expected reward plus the integral score
        loss = -((1 - alpha) * score + alpha * integral_score)

        # Print the loss after we have reached a multiple of 'print_iterations'
        if self.total_it % print_iterations == 0:
            print("Integral Score", alpha * integral_score.item())
            print("LCB Score:", (1 - alpha) * score.item())
            print(f"Actor Loss: {loss.item()}")
            print("")

        # Optimize the actor
        self.actor_optimizer.zero_grad()  # Clear old gradient
        loss.backward()  # Backpropagate the loss
        self.actor_optimizer.step()  # Update the parameters

    def get_integral_scores(self, beta, states, actions, step, max_delta, online=False):
        # Get the batch size and action dimension size from the passed actions
        batch_size, action_dim = actions.shape

        # Convert the "step" and "max_delta" lists to tensors and place on device
        step = torch.tensor(step, device=device)
        max_delta = torch.tensor(max_delta, device=device)

        # Get the number of steps for each action dimension
        num_steps = (max_delta / step).ceil()

        # Get the maximum step amount from array
        max_steps = int(num_steps.max().item())

        # Create a matrix of incrementally increasing values up to the max_steps value -> (max_steps)
        increments = torch.arange(1, max_steps + 1, device=device)

        # Generate a tensor of all increment steps in each action dimension
        # Convert "increments" tensor from (max_steps) -> (max_steps, 1)
        # Convert the "step" array from (action_dim) -> (1, action_dim)
        # Multiply the "increments" by "step" to get tensor of all steps in each dimension (m, 1) * (1, a) = (m, a)
        delta = increments.unsqueeze(1) * step.unsqueeze(0)

        # Clamp delta to max_delta
        delta = torch.minimum(delta, max_delta)

        # Define an identity matrix of size (action_dim, action_dim)
        identity = torch.eye(action_dim, device=device)

        # Get total perturbations for each action dimension
        # Convert "delta" tensor from (m, a) -> (m, a, 1)
        # Multiply "delta" by "identity" to get (m, a, 1) * (a, a) = (m, a, a)
        perturb = delta.unsqueeze(2) * identity

        # Convert the actions from (batch_size, action_dim) -> (batch_size, 1, 1, action_dim)
        actions_exp = actions.unsqueeze(1).unsqueeze(1)

        # Calculate the positively perturbed actions; (b, 1, 1, a) + (1, m, a, a) = (b, m, a, a)
        pos_actions = actions_exp + perturb.unsqueeze(0)

        # Calculate the negatively perturbed actions; (b, 1, 1, a) - (1, m, a, a) = (b, m, a, a)
        neg_actions = actions_exp - perturb.unsqueeze(0)

        # Combine the positively and negatively perturbed actions; (b, 2*m, a, a)
        all_actions = torch.cat([pos_actions, neg_actions], dim=1)

        # Flatten the actions above so that they can be passed through the reward model; (2*b*m*a, a)
        all_actions_flat = all_actions.reshape(-1, action_dim)

        # Generate a tensor of corresponding states for all actions
        # Convert "states" from (batch_size, state_dim) -> (batch_size, 1, 1, state_dim)
        # Expand "states" or duplicate entries to generate array of size (b, 2*m, a, s)
        # Convert the tensor from (b, 2*m, a, s) -> (2*b*m*a, s)
        states_flat = (
            states.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, 2 * max_steps, action_dim, -1)
            .reshape(batch_size * 2 * max_steps * action_dim, -1)
        )

        if online:
            # Get the scores of the actions
            scores = self.reward_model.reward_model(all_actions_flat)

        if not online:
            # Pass all of the perturbed actions with the corresponding states through the reward model
            means, stds = self.reward_model.reward_model.predict(states_flat, all_actions_flat)

            # Calculate the score (LCB) for all of the perturbed actions and convert from (b * a, 1) -> (b, a)
            scores = means - beta * stds

        # Convert scores from (2*b*m*a, 1) -> (b, 2*m, a)
        scores = scores.view(batch_size, 2 * max_steps, action_dim)

        # Create a mask of size (max_steps, action_dim) that stores booleans for valid increments
        valid_mask = (increments.unsqueeze(1) < num_steps.unsqueeze(0)).float()

        # Concatenate the valid mask with itself for negative and positive directions
        valid_mask = torch.cat([valid_mask, valid_mask], dim=0)

        # Apply the boolean mask to the scores
        scores = scores * valid_mask.unsqueeze(0)

        # Sum the scores across 2nd and 3rd dimensions (b, 2*m, a)
        score = scores.sum(dim=(1, 2))

        # Calculate the normalizer, which is twice the number of steps
        normalizer = 2 * num_steps.sum()

        # Calculate the normalized score for each action
        normalized_score = score / normalizer

        return normalized_score

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "/Policy")
        torch.save(self.actor_optimizer.state_dict(), filename + "/Policy_Optimizer")
        np.save(filename + "/Policy_Iterations.npy", self.total_it)

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "/Policy"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "/Policy_Optimizer"))
        self.total_it = int(np.load(filename + "/Policy_Iterations.npy"))


class A2P(object):
    def __init__(
        self,
        state_dim=4,
        action_dim=6,
        min_action=[-0.10, -0.10, -0.01, -0.7853981634, -0.7853981634, -0.7853981634],
        max_action=[0.10, 0.10, 0.10, 0.7853981634, 0.7853981634, 0.7853981634],
        expl_noise=[0.02, 0.02, 0.02, 0.10, 0.10, 0.10],
        noise_clip=[0.04, 0.04, 0.04, 0.20, 0.20, 0.20],
        model_path="~/RecycRL",
        initial_action=None,
    ):
        # Define NN for actor and adversarial actor, then define optimizers
        self.actor = Actor(state_dim, action_dim, min_action, max_action, initial_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.adversary = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.adversary_optimizer = torch.optim.Adam(self.adversary.parameters(), lr=3e-4)

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
        self.d = 0  # Tracking variable for moving average of distance between action and adversarial action
        self.epsilon = 0.1  # Variable for adversarial coefficient
        self.total_it = 0  # Tracking variable for number of iterations

        # If the reward model has been saved previously, load the model
        if model_path != "" and os.path.exists(f"{model_path}/Reward_Model"):
            self.reward_model.load(model_path)

    def select_action(self, state, add_noise=False, seed=0):
        # Convert the state to a row vector tensor and then add it to the GPU
        state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)

        # Pass the state through the actor network to get the action
        action = self.actor(state)

        # print("Action before noise", action.cpu().detach().numpy().flatten())

        # If the "add_noise" flag is set to True
        if add_noise:
            # Create a local noise generator and set the seed
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

            # Get random noise based on Gaussian with 0 mean and "expl_noise" variance with action size
            noise = torch.randn(action.shape, device=device, generator=generator)
            noise = (noise * self.expl_noise).clamp(-self.noise_clip, self.noise_clip)

            # print("Noise:", noise.cpu().detach().numpy().flatten())

            # Add the noise to the action
            action = action + noise

        # Clamp final action to valid bounds
        action = action.clamp(self.min_action, self.max_action)
        # action = torch.max(torch.min(action, self.max_action), self.min_action)

        return action.cpu().detach().numpy().flatten()

    def train(
        self,
        batch,
        alpha=0.50,
        beta=2.0,
        gamma=0.10,
        print_iterations=100,
    ):
        # Increment the iteration tracking variable
        self.total_it += 1

        # Extract the states from the passed batch
        states, _, _ = batch

        # Get the actor's actions for each state
        actions = self.actor(states)

        # Get the adversary's actions for each state
        adversarial_actions = self.adversary(states)

        # With gradient disabled
        with torch.no_grad():
            # Update the epsilon value based on the actions
            self.update_epsilon(actions, adversarial_actions, alpha, gamma)

        # Scale the actor and adversarial actions according to epsilon and combine, done twice to separate gradients
        combined_actions = actions * (1 - self.epsilon) + adversarial_actions.detach() * self.epsilon
        combined_adv_actions = actions.detach() * (1 - self.epsilon) + adversarial_actions * self.epsilon

        # Pass states and actor's actions through the reward model to get the mean and variance of the expected reward
        means, stds = self.reward_model.reward_model.predict(states, combined_actions)
        adv_means, adv_stds = self.reward_model.reward_model.predict(states, combined_adv_actions)

        # Get the scores of the actions which is the lower confidence bound
        scores = means - beta * stds
        adv_scores = adv_means - beta * adv_stds

        # Calculate the losses for the actor and adversary
        actor_loss = -(scores).mean()
        adversary_loss = adv_scores.mean()

        # Print the loss after we have reached a multiple of 'print_iterations'
        if self.total_it % print_iterations == 0:
            print(f"Actor Loss: {actor_loss.item()}")
            print(f"Adversary Loss: {adversary_loss.item()}")
            print("")

        # Optimize the actor and adversary
        self.actor_optimizer.zero_grad()  # Clear old gradient
        self.adversary_optimizer.zero_grad()  # Clear old gradient
        actor_loss.backward()  # Backpropagate the loss
        adversary_loss.backward()  # Backpropagate the loss
        self.actor_optimizer.step()  # Update the parameters
        self.adversary_optimizer.step()  # Update the parameters

    def train_for_buffalo(
        self,
        reward_model=None,
        online=True,
        alpha=0.5,
        beta=2.0,
        gamma=0.01,
        print_iterations=100,
    ):
        # Increment the iteration tracking variable
        self.total_it += 1

        # Define state as a tensor of size 1 with value to 0 since buffalo only has one state
        state = torch.tensor([[0.0]], device=device)

        # Get the actor's/policy's action
        action = self.actor(state)

        # Get the adversary's action for the state
        adversarial_action = self.adversary(state)

        # With gradient disabled
        with torch.no_grad():
            # Update the epsilon value based on the actions
            self.update_epsilon(action, adversarial_action, alpha, gamma)

        # Scale the actor and adversarial actions according to epsilon and combine, done twice to separate gradients
        combined_action = action * (1 - self.epsilon) + adversarial_action.detach() * self.epsilon
        combined_adv_action = action.detach() * (1 - self.epsilon) + adversarial_action * self.epsilon

        # If we are training a policy online
        if online:
            # Assign the passed reward model to the policy's reward model
            self.reward_model.reward_model = reward_model.to(device)

            # Get the score of the action
            score = self.reward_model.reward_model(combined_action)
            adv_score = self.reward_model.reward_model(combined_adv_action)

        # If we are training a policy offline with a learned reward model
        elif not online:
            # Pass state and each actors' action through the reward model to get the mean and variance of the expected reward
            mean, std = self.reward_model.reward_model.predict(state, combined_action)
            adv_mean, adv_std = self.reward_model.reward_model.predict(state, combined_adv_action)

            # Get the scores of the actions which is the lower confidence bound
            score = mean - beta * std
            adv_score = adv_mean - beta * adv_std

        # Calculate the losses for the actor and adversary
        actor_loss = -(score)
        adversary_loss = adv_score

        # Print the loss after we have reached a multiple of 'print_iterations'
        if self.total_it % print_iterations == 0:
            print(f"Actor Loss: {actor_loss.item()}")
            print(f"Adversary Loss: {adversary_loss.item()}")
            print("")

        # Optimize the actor and adversary
        self.actor_optimizer.zero_grad()  # Clear old gradient
        self.adversary_optimizer.zero_grad()  # Clear old gradient
        actor_loss.backward()  # Backpropagate the loss
        adversary_loss.backward()  # Backpropagate the loss
        self.actor_optimizer.step()  # Update the parameters
        self.adversary_optimizer.step()  # Update the parameters

    def update_epsilon(self, actions, adversarial_actions, alpha, gamma):
        # Calculate the difference between the actor and adversarial actions
        action_diff = actions - adversarial_actions

        # Get the norm of the vectors to calculate the distance between the two actions, then average
        dist = torch.norm(action_diff).mean().item()

        # Recompute the new d with the moving average
        new_d = alpha * self.d + (1 - alpha) * dist

        # Calculate the difference between the current distance and previous distance
        dist_diff = new_d - self.d

        # Get the sign of the distance difference
        sign = np.sign(dist_diff)

        # Get the magnitude of the distance difference using a sigmoid function
        magnitude = torch.sigmoid(abs(torch.tensor(dist_diff))).item()

        # Calculate the new epsilon value by subtracting the product of the sign and magnitude from the previous epsilon value
        new_epsilon = (self.epsilon - gamma * sign * magnitude).clip(0.03, 0.20)

        # Update d and epsilon values
        self.d = new_d
        self.epsilon = new_epsilon

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "/Policy_A2P")
        torch.save(self.actor_optimizer.state_dict(), filename + "/Policy_A2P_Optimizer")
        np.save(filename + "/Policy_A2P_Distance.npy", self.d)
        np.save(filename + "/Policy_A2P_Epsilon.npy", self.epsilon)
        np.save(filename + "/Policy_A2P_Iterations.npy", self.total_it)

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "/Policy_A2P"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "/Policy_A2P_Optimizer"))
        self.d = float(np.load(filename + "/Policy_A2P_Distance.npy"))
        self.epsilon = float(np.load(filename + "/Policy_A2P_Epsilon.npy"))
        self.total_it = int(np.load(filename + "/Policy_A2P_Iterations.npy"))


class NRMDP(object):
    def __init__(
        self,
        state_dim=4,
        action_dim=6,
        min_action=[-0.10, -0.10, -0.01, -0.7853981634, -0.7853981634, -0.7853981634],
        max_action=[0.10, 0.10, 0.10, 0.7853981634, 0.7853981634, 0.7853981634],
        expl_noise=[0.02, 0.02, 0.02, 0.10, 0.10, 0.10],
        noise_clip=[0.04, 0.04, 0.04, 0.20, 0.20, 0.20],
        model_path="~/RecycRL",
        epsilon=0.1,
        initial_action=None,
    ):
        # Define NN for actor and adversarial actor, then define optimizers
        self.actor = Actor(state_dim, action_dim, min_action, max_action, initial_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.adversary = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.adversary_optimizer = torch.optim.Adam(self.adversary.parameters(), lr=3e-4)

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
        self.epsilon = epsilon  # Variable for adversarial coefficient
        self.total_it = 0  # Tracking variable for number of iterations
        self.prev_rewards = []  # Tracking variable for previous rewards

        # If the reward model has been saved previously, load the model
        if model_path != "" and os.path.exists(f"{model_path}/Reward_Model"):
            self.reward_model.load(model_path)

    def select_action(self, state, add_noise=False, seed=0):
        # Convert the state to a row vector tensor and then add it to the GPU
        state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)

        # Pass the state through the actor network to get the action
        action = self.actor(state)

        # print("Action before noise", action.cpu().detach().numpy().flatten())

        # If the "add_noise" flag is set to True
        if add_noise:
            # Create a local noise generator and set the seed
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

            # Get random noise based on Gaussian with 0 mean and "expl_noise" variance with action size
            noise = torch.randn(action.shape, device=device, generator=generator)
            noise = (noise * self.expl_noise).clamp(-self.noise_clip, self.noise_clip)

            # print("Noise:", noise.cpu().detach().numpy().flatten())

            # Add the noise to the action
            action = action + noise

        # Clamp final action to valid bounds
        action = action.clamp(self.min_action, self.max_action)
        # action = torch.max(torch.min(action, self.max_action), self.min_action)

        return action.cpu().detach().numpy().flatten()

    def train(
        self,
        batch,
        beta=2.0,
        print_iterations=100,
    ):
        # Increment the iteration tracking variable
        self.total_it += 1

        # Extract the states from the passed batch
        states, _, _ = batch

        # Get the actor's actions for each state
        actions = self.actor(states)

        # Get the adversary's actions for each state
        adversarial_actions = self.adversary(states)

        # Scale the actor and adversarial actions according to epsilon and combine, done twice to separate gradients
        combined_actions = actions * (1 - self.epsilon) + adversarial_actions.detach() * self.epsilon
        combined_adv_actions = actions.detach() * (1 - self.epsilon) + adversarial_actions * self.epsilon

        # Pass states and actor's actions through the reward model to get the mean and variance of the expected reward
        means, stds = self.reward_model.reward_model.predict(states, combined_actions)
        adv_means, adv_stds = self.reward_model.reward_model.predict(states, combined_adv_actions)

        # Get the scores of the actions which is the lower confidence bound
        scores = means - beta * stds
        adv_scores = adv_means - beta * adv_stds

        # Calculate the losses for the actor and adversary
        actor_loss = -(scores).mean()
        adversary_loss = adv_scores.mean()

        # Print the loss after we have reached a multiple of 'print_iterations'
        if self.total_it % print_iterations == 0:
            print(f"Actor Loss: {actor_loss.item()}")
            print(f"Adversary Loss: {adversary_loss.item()}")
            print("")

        # Optimize the actor and adversary
        self.actor_optimizer.zero_grad()  # Clear old gradient
        self.adversary_optimizer.zero_grad()  # Clear old gradient
        actor_loss.backward()  # Backpropagate the loss
        adversary_loss.backward()  # Backpropagate the loss
        self.actor_optimizer.step()  # Update the parameters
        self.adversary_optimizer.step()  # Update the parameters

    def train_for_buffalo(
        self,
        reward_model=None,
        online=True,
        beta=2.0,
        print_iterations=100,
    ):
        # Increment the iteration tracking variable
        self.total_it += 1

        # Define state as a tensor of size 1 with value to 0 since buffalo only has one state
        state = torch.tensor([[0.0]], device=device)

        # Get the actor's/policy's action
        action = self.actor(state)

        # Get the adversary's action for the state
        adversarial_action = self.adversary(state)

        # Scale the actor and adversarial actions according to epsilon and combine, done twice to separate gradients
        combined_action = action * (1 - self.epsilon) + adversarial_action.detach() * self.epsilon
        combined_adv_action = action.detach() * (1 - self.epsilon) + adversarial_action * self.epsilon

        # If we are training a policy online
        if online:
            # Assign the passed reward model to the policy's reward model
            self.reward_model.reward_model = reward_model.to(device)

            # Get the score of the action
            score = self.reward_model.reward_model(combined_action)
            adv_score = self.reward_model.reward_model(combined_adv_action)

        # If we are training a policy offline with a learned reward model
        elif not online:
            # Pass state and each actors' action through the reward model to get the mean and variance of the expected reward
            mean, std = self.reward_model.reward_model.predict(state, combined_action)
            adv_mean, adv_std = self.reward_model.reward_model.predict(state, combined_adv_action)

            # Get the scores of the actions which is the lower confidence bound
            score = mean - beta * std
            adv_score = adv_mean - beta * adv_std

        # Calculate the losses for the actor and adversary
        actor_loss = -(score)
        adversary_loss = adv_score

        # Print the loss after we have reached a multiple of 'print_iterations'
        if self.total_it % print_iterations == 0:
            print(f"Actor Loss: {actor_loss.item()}")
            print(f"Adversary Loss: {adversary_loss.item()}")
            print("")

        # Optimize the actor and adversary
        self.actor_optimizer.zero_grad()  # Clear old gradient
        self.adversary_optimizer.zero_grad()  # Clear old gradient
        actor_loss.backward()  # Backpropagate the loss
        adversary_loss.backward()  # Backpropagate the loss
        self.actor_optimizer.step()  # Update the parameters
        self.adversary_optimizer.step()  # Update the parameters

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "/Policy_NRMDP")
        torch.save(self.actor_optimizer.state_dict(), filename + "/Policy_NRMDP_Optimizer")
        np.save(filename + "/Policy_NRMDP_Iterations.npy", self.total_it)

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "/Policy_NRMDP"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "/Policy_NRMDP_Optimizer"))
        self.total_it = int(np.load(filename + "/Policy_NRMDP_Iterations.npy"))
