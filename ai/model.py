# Save this as model.py under the ai directory
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from matplotlib import pyplot as plt
import numpy as np
from ai.replay_memory import PrioritizedReplayMemory
import torch.nn.functional as F
from collections import deque

class AIPlayer:
    def __init__(self, game, model_path=None, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, tau=0.01, batch_size=32, lr=0.001):
        self.game = game
        self.model = DuelingCNN(input_channels=1, board_size=game.grid_size)  # Dueling Q-network
        self.target_model = DuelingCNN(input_channels=1, board_size=game.grid_size)  # Dueling Target Network
        self.target_model.load_state_dict(self.model.state_dict())  # Make sure both models have the same initial weights
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)  # Decrease learning rate every 1000 steps

        self.loss_fn = torch.nn.SmoothL1Loss()

        self.memory = PrioritizedReplayMemory(10000)
        self.target_update = 1000
        self.last_saved_performance = -np.inf
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size

        # self.batch_size = 50
        self.update_every = 1000  # how often to update the target network
        self.step_counter = 0  # counter to keep track of steps

        self.epsilon_increase = 0.01  # The amount by which we increase epsilon when performance degrades
        self.validation_states = []  # A list to store validation states
        self.validation_rewards = []  # A list to store the model's performance on the validation states

        # UCB specific initialization
        self.action_counts = np.zeros(game.grid_size)  # Track how many times each action was taken
        self.total_action_values = np.zeros(game.grid_size)  # Sum of values received from each action
        self.ucb_c = 4  # Exploration parameter for UCB, tune this based on how much you value exploration

        self.episode_durations = []
        self.episode_rewards = []
        self.epsilon_values = []  
        self.lr_values = []  
        self.q_values = []  
        self.game_results = []

        # Load model if a path is provided
        if model_path is not None:
            self.load_model(model_path)

    def update_rewards(self, reward):
        self.memory.update_rewards(reward)

    def remember(self, state, action, reward, next_state):
        game_over = self.game.is_game_over()
        self.memory.push((state, action, reward, next_state, game_over))

    def get_random_valid_move(self):
        move = random.randrange(self.game.grid_size)
        while not self.game.is_valid_move(move):
            move = random.randrange(self.game.grid_size)
        return move
    
    def record_game_result(self, game_result):
        self.game_results.append(game_result)

    def choose_action(self, state, episode):
        self.epsilon_values.append(self.epsilon)  
        self.lr_values.append(self.optimizer.param_groups[0]['lr'])
        
        # Calculate Q-values first
        with torch.no_grad():
            state = np.array(state)
            if state.ndim == 2:
                state = np.expand_dims(state, axis=0)  # Add batch dimension
                state = np.expand_dims(state, axis=0)  # Add channel dimension
            else:
                raise ValueError("State shape is not compatible with the CNN model.")

            q_values = self.model(torch.tensor(state, dtype=torch.float32)).cpu().numpy()
            
            # Use the new get_valid_moves method to consider symmetrical states
            valid_moves = self.game.get_valid_moves()

            # Log the maximum Q-value for the current state
            max_q_value = np.max(q_values[0, valid_moves])  # Only consider valid moves
            self.q_values.append(max_q_value)  # Append to q_values for logging

        # Apply UCB to valid moves
        ucb_values = np.zeros(self.game.grid_size)
        total_counts = np.sum(self.action_counts)
        for move in valid_moves:
            if self.action_counts[move] > 0:
                avg_value = self.total_action_values[move] / self.action_counts[move]
                ucb_bonus = self.ucb_c * np.sqrt(np.log(total_counts) / self.action_counts[move])
                ucb_values[move] = avg_value + ucb_bonus
            else:
                ucb_values[move] = np.inf  # Assign infinity if the move has never been tried

        # Choose action based on UCB values
        move = np.argmax(ucb_values)
        
        # Update action counts and total action values
        self.action_counts[move] += 1
        self.total_action_values[move] += q_values[0][move]  # Assuming q_values is a 2D array with shape (1, grid_size)

        # Record the Q-value of the chosen action
        self.q_values[-1] = q_values[0][move]  # Update the last logged Q-value with the Q-value of the chosen action

        # Adjust epsilon after choosing action
        self.adjust_epsilon(episode)
        return move

       
    def update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def evaluate(self):
        """
        Evaluate the model's performance on a separate validation environment.
        """
        if len(self.validation_states) == 0:  # If we don't have any validation states yet
            return

        total_reward = 0
        for state in self.validation_states:  # For each validation state
            with torch.no_grad():
                # Flatten the state before passing to model and add an extra dimension for batch size
                flattened_state = np.array(state).reshape(1, -1)
                q_values = self.model(torch.tensor(flattened_state).float())
                total_reward += torch.max(q_values).item()  # Add the max Q-value to the total reward

        avg_reward = total_reward / len(self.validation_states)  # Calculate the average reward
        self.validation_rewards.append(avg_reward)  # Add it to the list of validation rewards

        print(f"Average validation reward: {avg_reward}")

    def train(self):
        try:
            if len(self.memory) < self.batch_size:
                return
            batch, indices, weights = self.memory.sample(self.batch_size)
            state, action, reward, next_state, game_over = zip(*batch)

            # Reshape for CNN input: (batch_size, channels, height, width)
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(1)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(1)

            action = torch.tensor(action).long()
            reward = torch.tensor(reward).float()
            game_over = torch.tensor(game_over if game_over is not None else 0).float()
            weights = torch.tensor(weights)

            # Get current Q estimates
            current_q_values = self.model(state).gather(1, action.unsqueeze(1))

            # Compute the Q value target using Double DQN Update
            with torch.no_grad():
                next_state_actions = self.model(next_state).max(1)[1]
                next_state_values = self.target_model(next_state).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
            expected_q_values = reward + (1 - game_over) * self.gamma * next_state_values

            # Calculate the TD error
            td_error = (expected_q_values - current_q_values.squeeze()).detach().abs().numpy()
            # Update sample priorities
            self.memory.update_priorities(indices, td_error + 1e-6)

            # Multiply the loss by the importance sampling weights
            loss = (weights * self.loss_fn(current_q_values, expected_q_values.unsqueeze(1))).mean()

            self.optimizer.zero_grad()
            loss.backward()
            # Apply gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.lr_scheduler.step()

            # Add the current loss and reward to the logging lists.
            self.episode_rewards.append(np.mean(reward.numpy()))
            self.episode_durations.append(loss.item())

            # Increment step counter
            self.step_counter += 1

            # Update the target network periodically
            if self.step_counter % self.update_every == 0:
                self.update_target_model()

            # Perform validation periodically
            if self.step_counter % 100 == 0:
                self.evaluate_on_validation_states()

        except Exception as e:
            self.game.logger.error(f"Exception occurred during training: {str(e)}")
            return None


    def evaluate_on_validation_states(self):
        if not self.validation_states:
            return
        total_reward = 0
        for state in self.validation_states:
            # Evaluate the model on the state and calculate reward
            # For simplicity, let's assume a function `calculate_reward` does this
            reward = self.calculate_reward(state)
            total_reward += reward
        avg_reward = total_reward / len(self.validation_states)
        self.validation_rewards.append(avg_reward)
        print(f"Validation Avg Reward: {avg_reward}")

    def adjust_epsilon(self, episode):
        """
        Adjust epsilon based on the episode number and recent performance.
        Epsilon will decrease over time, reducing the exploration rate and increasing exploitation.
        """
        if len(self.episode_rewards) > 100:  # Ensure we have enough episodes to evaluate performance
            avg_reward = np.mean(self.episode_rewards[-100:])  # Average reward over the last 100 episodes

            # Dynamic decay rate adjustment based on performance
            performance_factor = max(0.5, 2 - avg_reward / self.last_saved_performance) if self.last_saved_performance != -np.inf else 1.0

            # Adjust epsilon based on the dynamic decay rate
            self.epsilon *= self.epsilon_decay ** performance_factor

            # Optionally, factor in performance improvement
            if avg_reward > self.last_saved_performance:  # If performance improved
                self.last_saved_performance = avg_reward  # Update the last saved performance
            else:  # If performance degraded
                self.epsilon = min(1.0, self.epsilon + self.epsilon_increase)  # Increase epsilon
        else:
            # If not enough episodes, fall back to exponential decay
            self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-0.01 * episode)

        # Ensure epsilon is within the defined bounds
        self.epsilon = max(self.epsilon_min, min(1.0, self.epsilon))

        # Log epsilon adjustments for monitoring
        self.epsilon_values.append(self.epsilon)


    def save_model(self, path, episode):
        try:
            if len(self.episode_rewards) > 0:  # Check if the list is not empty
                torch.save({
                    'episode': episode,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'loss': self.loss_fn,
                    'epsilon': self.epsilon,
                }, path)
                print(f"Model saved at episode {episode}")
        except Exception as e:
            self.game.logger.error(f"Exception occurred while saving model: {str(e)}")

    def save_best_model(self, path, episode):
        try:
            if len(self.episode_rewards) > 0:  # Check if the list is not empty
                avg_reward = np.mean(self.episode_rewards[-100:])
                self.game.logger.info(f"Average reward for the last 100 episodes: {avg_reward}, Last saved performance: {self.last_saved_performance}")
                print(path)

                if avg_reward > self.last_saved_performance:
                    torch.save({
                        'episode': episode,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.lr_scheduler.state_dict(),
                        'loss': self.loss_fn,
                        'epsilon': self.epsilon,
                    }, path)
                    self.last_saved_performance = avg_reward
                    print(f"Best model saved at episode {episode} with avg reward {avg_reward} to path '{path}'")
                else:
                    print(f"No improvement in performance at episode {episode}. Model not saved. Intended path was '{path}'.")
        except Exception as e:
            self.game.logger.error(f"Exception occurred while saving best model at '{path}': {str(e)}")



    def load_model(self, path):
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.loss_fn = checkpoint['loss']
            self.epsilon = checkpoint['epsilon']
            self.game_results = checkpoint['game_results']
            episode = checkpoint['episode']
            return episode
        except Exception as e:
            self.game.logger.error(f"Exception occurred while loading model: {str(e)}")
            return None
        
    def plot_durations(self):
        plt.figure(figsize=(15,10))

        plt.subplot(2,2,1)
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(self.episode_durations)

        plt.subplot(2,2,2)
        plt.title('Training Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(self.episode_rewards)

        plt.subplot(2,2,3)  
        plt.title('Epsilon')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.plot(self.epsilon_values)

        plt.subplot(2,2,4)  
        plt.title('Learning Rate')
        plt.xlabel('Episode')
        plt.ylabel('Learning Rate')
        plt.plot(self.lr_values)

        if len(self.episode_durations) >= 100:  # Check if the number of episodes is greater than or equal to 100
            if len(self.episode_durations) % 100 == 0:  # Check if the number of episodes is divisible by 100
                means = np.array(self.episode_durations).reshape(-1, 100).mean(1)
                plt.plot(np.arange(0, len(self.episode_durations), 100), means)

        plt.show()

    def plot_q_values(self):  
        plt.figure(figsize=(10,7))
        plt.title('Q-Values')
        plt.xlabel('Step')
        plt.ylabel('Q-Value')
        plt.plot(self.q_values)
        plt.show()

    def plot_validation_rewards(self):
        """
        Plot the model's performance on the validation environment.
        """
        plt.figure(figsize=(10, 7))
        plt.title('Validation Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(self.validation_rewards)
        plt.show()
        
    def record_performance(self):
        wins, losses, ties = 0, 0, 0
        for game_result in self.game_results:
            if game_result == 'win':
                wins += 1
            elif game_result == 'loss':
                losses += 1
            elif game_result == 'draw':
                ties += 1
                
        if len(self.game_results) == 0:
            self.game.logger.info('No games played yet.')
            return

        win_rate = wins / len(self.game_results)
        loss_rate = losses / len(self.game_results)
        tie_rate = ties / len(self.game_results)
        
        self.game.logger.info(f"Win rate: {win_rate:.2f}, Loss rate: {loss_rate:.2f}, Draw rate: {tie_rate:.2f}")

    def get_game_results(self):
        wins = self.game_results.count('win')
        losses = self.game_results.count('loss')
        draws = self.game_results.count('draw')
        return wins, losses, draws
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return F.relu(out)
    
class DuelingCNN(nn.Module):
    def __init__(self, input_channels=1, board_size=6):
        super(DuelingCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.resblock1 = ResidualBlock(64)
        self.resblock2 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Value stream
        self.fc1_val = nn.Linear(128 * board_size * board_size, 128)
        self.fc2_val = nn.Linear(128, 1)

        # Advantage stream
        self.fc1_adv = nn.Linear(128 * board_size * board_size, 128)
        self.fc2_adv = nn.Linear(128, board_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor

        val = F.relu(self.fc1_val(x))
        val = self.fc2_val(val)

        adv = F.relu(self.fc1_adv(x))
        adv = self.fc2_adv(adv)

        # Combine streams
        return val + adv - adv.mean(1, keepdim=True)
