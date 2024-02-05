# Save this as engine.py under the game directory
from ai.model import AIPlayer
import os
import pygame
import numpy as np
import logging
import time
import random
import traceback

class HumanPlayer:
    def __init__(self, game):
        self.game = game

    def choose_action(self, column, state, epsilon=None):
        return column

class Game:
    MAX_RETRIES = 5  # Max retries for invalid moves

    def __init__(self, grid_size=6, win_length=4, reward_win=1.0, reward_draw=0.5, reward_loss=-1.0, reward_move=-0.01):
        self.grid_size = grid_size
        self.win_length = win_length
        self.reward_win = reward_win
        self.reward_draw = reward_draw
        self.reward_loss = reward_loss
        self.reward_move = reward_move
        # Add an attribute to track the previous state
        self.previous_state = None
        self.ai_player = AIPlayer(self)  # Default AI player is AIPlayer instead of RandomAIPlayer
        self.ai_player2 = AIPlayer(self)  # Second AI player
        self.winner = None  # Add this line
        self.starting_player = 1  # Player 1 starts first
        self.current_player = self.starting_player
        self.last_move = None
        self.human_player = HumanPlayer(self)  # Human player
        self.is_human_playing = False  # Add this flag
        self.game_running = False  # Game is not running by default
        self.game_paused = False  # Game is not paused by default
        self.loser = None  # Player who lost the last game
        self.ai_wins = 0  # Number of wins by the AI player
        self.ai2_wins = 0
        self.human_wins = 0  # Number of wins by the human player
        
        self.reset()
        # Moved logging setup to separate method
        self.setup_logger()
        # Moved performance metrics initialization to separate method
        self.init_metrics()
        self._check_reward_conditions(reward_win, reward_draw, reward_loss, reward_move)

        if not isinstance(grid_size, int) or grid_size <= 0:
            raise ValueError("Grid size must be a positive integer.")
        if not isinstance(win_length, int) or win_length <= 0 or win_length > grid_size:
            raise ValueError("Win length must be a positive integer no greater than grid size.")
        if not isinstance(reward_win, (int, float)) or not isinstance(reward_draw, (int, float)) or not isinstance(reward_loss, (int, float)) or not isinstance(reward_move, (int, float)):
            raise ValueError("All rewards must be numeric values.")
        if not isinstance(reward_win, (int, float)) or reward_win < reward_draw or reward_win < reward_loss or reward_win < reward_move:
            raise ValueError("Win reward must be greater than draw, loss, and move rewards.")
        
    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('game.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def init_metrics(self):
        self.start_time = time.time()
        self.end_time = None

    def _check_reward_conditions(self, reward_win, reward_draw, reward_loss, reward_move):
        if not isinstance(reward_win, (int, float)) or not isinstance(reward_draw, (int, float)) or not isinstance(reward_loss, (int, float)) or not isinstance(reward_move, (int, float)):
            raise ValueError("All rewards must be numeric values.")
        if not isinstance(reward_win, (int, float)) or reward_win < reward_draw or reward_win < reward_loss or reward_win < reward_move:
            raise ValueError("Win reward must be greater than draw, loss, and move rewards.")

    def update_rewards(self, reward_win, reward_draw, reward_loss, reward_move):

        self._check_reward_conditions(reward_win, reward_draw, reward_loss, reward_move)
        self.reward_win = reward_win
        self.reward_draw = reward_draw
        self.reward_loss = reward_loss
        self.reward_move = reward_move

    def toggle_human_play(self, gui, model_path=None):
        self.is_human_playing = not self.is_human_playing
        if self.is_human_playing:
            print("Human Play mode activated")
            self.game_running = True  # Ensure the game is marked as running
            if model_path:
                self.play_against_ai(gui, model_path)
            else:
                print("AI model path not provided. Loading default AI.")
                self.play_human(gui)
        else:
            print("AI vs AI2 mode activated")
            self.game_running = False  # Stop the game if exiting human play

    def start_game(self):
        self.game_running = True
        self.game_paused = False
    

    def pause_game(self):
        self.game_paused = True
        print("Game paused")

    def resume_game(self):
        self.game_paused = False
        print("Game resumed")

        
    def shape_reward(self, state, action, reward, next_state, game_over, player):
        # Start with the base reward logic
        if game_over:
            if self.winner == player:
                return self.reward_win
            else:
                return self.reward_loss
        else:
            reward = self.reward_move  # Default move reward

        # Integrate the additional reward shaping conditions
        reward += 0.2 * self.blocks_opponent_win(state, next_state, player)  # Reward for blocking opponent's win
        reward += 0.3 * self.creates_multiple_opportunities(next_state, player)  # Reward for creating multiple win opportunities

        # Penalty if the move leads to an opponent's win
        if self.move_leads_to_opponent_win(next_state, player):
            reward -= 0.5  # Increased penalty for leading the opponent to win

        # Penalize stagnation (no significant change in state)
        if np.array_equal(self.previous_state, next_state):
            reward -= 0.1  # Increased penalty for stagnation

        self.previous_state = next_state
        return reward


    
    def move_leads_to_opponent_win(self, next_state, player):
        opponent = 2 if player == 1 else 1
        for move in self.get_valid_moves():
            # Simulate the opponent's move
            hypothetical_next_state = next_state.copy()
            hypothetical_next_state[move] = opponent

            # Temporarily set the board to the hypothetical state
            original_state = self.board.copy()
            self.board = hypothetical_next_state
            game_over = self.is_game_over()
            winner = self.winner

            # Restore the original state
            self.board = original_state

            # Check if the opponent has a winning move
            if game_over and winner == opponent:
                return True
        return False
    
    def blocks_opponent_win(self, state, next_state, player):
        opponent = 2 if player == 1 else 1
        if self.get_max_sequence_length(state, opponent) < self.win_length - 1 and self.get_max_sequence_length(next_state, opponent) == self.win_length - 1:
            return True
        return False

    def creates_multiple_opportunities(self, state, player):
        # This function checks if there are multiple lines where the player only needs one more move to win.
        count_opportunities = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if state[row][col] == player:
                    for dx, dy in directions:
                        sequence_length = 0
                        empty_spots = 0
                        for diff in range(self.win_length):
                            x, y = row + dx * diff, col + dy * diff
                            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                                if state[x][y] == player:
                                    sequence_length += 1
                                elif state[x][y] == 0:
                                    empty_spots += 1
                        if sequence_length == self.win_length - 1 and empty_spots == 1:
                            count_opportunities += 1

        return count_opportunities >= 2  # Return true if there are two or more opportunities to win
    
    def is_symmetrical(self):
        """
        Check if the current board state is symmetrical.
        """
        mid = self.grid_size // 2
        for row in range(self.grid_size):
            for col in range(mid):
                if self.board[row][col] != self.board[row][self.grid_size - 1 - col]:
                    return False
        return True

    def get_valid_moves(self):
        """
        Get a list of valid moves. If the state is symmetrical,
        only consider half of the columns plus the middle one if grid_size is odd.
        """
        if self.is_symmetrical():
            mid = self.grid_size // 2
            valid_moves = [c for c in range(mid + 1) if self.is_valid_move(c)]
        else:
            valid_moves = [c for c in range(self.grid_size) if self.is_valid_move(c)]
        return valid_moves
        
    def position_improved(self, state, next_state, player):
        # Position is considered improved if we have a longer sequence after the move or we're closer to winning.

        sequence_before = self.get_max_sequence_length(state, player)
        sequence_after = self.get_max_sequence_length(next_state, player)
        return sequence_after > sequence_before
    
    def position_worsened(self, state, next_state, player):
        # Position is considered worsened if opponent has a longer sequence after our move or if they're closer to winning.

        opponent = 2 if player == 1 else 1
        sequence_before = self.get_max_sequence_length(state, opponent)
        sequence_after = self.get_max_sequence_length(next_state, opponent)
        return sequence_after > sequence_before
    
    def get_max_sequence_length(self, state, player):
        # This function returns the length of the longest sequence of the player's pieces on the board.

        directions = [(0,1), (1,0), (1,1), (1,-1)]
        max_sequence_length = 0

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if state[row][col] == player:
                    for dx, dy in directions:
                        try:
                            sequence_length = sum(1 for diff in range(self.win_length) if state[row + dx * diff][col + dy * diff] == player)
                            max_sequence_length = max(max_sequence_length, sequence_length)
                        except IndexError:
                            pass

        return max_sequence_length
    
    def reset(self):
        self.board = np.zeros((self.grid_size, self.grid_size))
        self.current_player = self.starting_player  

    def is_draw(self):
        if not (self.board == 0).any() and not self.is_game_over():
            print("Game is a draw.")
            return True
        return False
    
    def get_game_result(self):
        game_result = {}
        if self.is_game_over():
            game_result["winner"] = self.winner
            game_result["draw"] = False
        elif self.is_draw():
            game_result["winner"] = None
            game_result["draw"] = True
        else:
            game_result["winner"] = None
            game_result["draw"] = False
        return game_result

    def load_model(self, ai_path=None):
        if ai_path is not None and os.path.exists(ai_path):
            self.ai_player = AIPlayer(self)
            self.ai_player.load_model(ai_path)

        else:
            self.ai_player = AIPlayer(self)
        self.human_player = HumanPlayer(self)

    def is_valid_move(self, column):
        return self.board[0][column] == 0
    
    def get_reward(self, player):
        if self.is_game_over():
            return self.reward_win if self.winner == player else self.reward_loss
        elif self.is_draw():
            return self.reward_draw
        else:
            return self.reward_move # Return the move reward if the game is still ongoing

    def get_valid_moves(self):
        return [c for c in range(self.grid_size) if self.is_valid_move(c)]

    def make_move(self, column, player):

        if not self.is_valid_move(column):
            raise ValueError(f"Invalid move made. Column: {column} is already full.")
        if not isinstance(player, int) or player not in [1, 2]:
            raise ValueError("Player must be either 1 or 2.")
        
        for row in reversed(range(self.grid_size)):
            if self.board[row][column] == 0:
                self.board[row][column] = player
                break
        if self.is_game_over():
            self.winner = player  # Set the winner

        return {"is_over": self.is_game_over(), "winner": self.winner, "is_draw": self.is_draw()}  # return game status

    def get_state(self):
        # Normalize the board state
        normalized_state = np.where(self.board == 1, -1, self.board)  # Map player 1 to -1
        normalized_state = np.where(normalized_state == 2, 1, normalized_state)  # Map player 2 to 1
        return normalized_state.copy()

    def remember(self, state, action, reward, next_state):
        self.ai_player.memory.push((state, action, reward, next_state))
        self.ai_player2.memory.push((state, action, reward, next_state))


    def is_game_over(self):
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.board[row][col] != 0:
                    for dx, dy in directions:
                        try:
                            if all(self.board[row + dx * diff][col + dy * diff] == self.board[row][col] for diff in range(self.win_length)):
                                print(f"Game Over! Player {self.board[row][col]} wins.")
                                return True
                        except IndexError:
                            pass
        return False

    def play_human(self, gui):
        self.reset()
        while gui.running:
            
            if self.current_player == 1:  # AI's turn
                state = self.get_state()
                action = self.ai_player.choose_action(state, 0)
                self.make_move(action, self.current_player)
                gui.draw_board()
            else:  # Human's turn
                action = gui.get_player_input()
                if action is not None and action in self.get_valid_moves():
                    self.make_move(action, self.current_player)
                    gui.draw_board()

            if self.is_game_over() or self.is_draw():
                print("Debug: Game Over or Draw detected")
                gui._display_game_over_message()
                break

            self.current_player = 2 if self.current_player == 1 else 1
            gui._handle_events()

        # Continue to process GUI events after game over
        while gui.running:
            gui._handle_events()
            gui.manager.update(pygame.time.Clock().tick(60) / 1000.0)
            pygame.display.update()

    def play_against_ai(self, gui, model_path):
        self.load_model(model_path)  # Load the best-trained AI model
        self.is_human_playing = True
        self.reset()
        while gui.running:
            if self.current_player == 1:  # AI's turn
                state = self.get_state()
                action = self.ai_player.choose_action(state, 0)
                self.make_move(action, self.current_player)
                gui.draw_board()
            else:  # Human's turn
                action = gui.get_player_input()
                if action is not None and action in self.get_valid_moves():
                    self.make_move(action, self.current_player)
                    gui.draw_board()

            if self.is_game_over() or self.is_draw():
                print("Game Over or Draw detected")
                gui._display_game_over_message()
                break

            self.current_player = 2 if self.current_player == 1 else 1
            gui._handle_events()

        # Continue to process GUI events after game over
        while gui.running:
            gui._handle_events()
            gui.manager.update(pygame.time.Clock().tick(60) / 1000.0)
            pygame.display.update()

    def play(self, gui=None, episodes=1000, batch_size=50, model_path='ai_model.pth'):

        if not isinstance(episodes, int) or episodes <= 0:
            raise ValueError("Episodes must be a positive integer.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if model_path is not None and not isinstance(model_path, str):
            raise ValueError("Model path must be a string.")
            
        self.draws = 0  # Initialize draws to 0
        # If there was a loser in the last game, they start the next game
        # self.current_player = self.loser if self.loser else random.choice([1, 2])
        
        for episode in range(episodes):
            self.start_game()  # Ensure the game starts in a running state
            try:
                if gui.is_human_playing:
                    # Logic for playing against human
                    self.play_human(gui)
                    continue  # Skip the rest of the loop to not train AI during human play

                self.reset()
                state = self.get_state()
                # self.current_player = self.starting_player
                # self.starting_player = 2 if self.starting_player == 1 else 1

                while self.game_running:

                    if gui:
                        gui._handle_events()  # Make sure to process GUI events

                    if self.game_paused:
                        pygame.time.wait(100)  # Sleep to prevent high CPU usage, adjust time as needed
                        continue  # Skip this iteration of the loop

                    if self.current_player == 1:  # AI player
                        action = self.ai_player.choose_action(state, episode)
                    else:  # AI player 2
                        action = self.ai_player2.choose_action(state, episode)
                        
                    self.make_move(action, self.current_player)
                    next_state = self.get_state()
                    game_over = self.is_game_over()
                    reward = self.shape_reward(state, action, self.get_reward(self.current_player), next_state, game_over, self.current_player)  # use shape_reward instead of get_reward
                    
                    if self.current_player == 1:  # AI player 1
                        self.ai_player.remember(state, action, reward, next_state)  # Remember the reward for the AI player 1
                        self.logger.info(f"AI Player 1 chose action {action}, new state: {state}")
                    else:  # AI player 2
                        self.ai_player2.remember(state, action, reward, next_state)  # Remember the reward for the AI player 2
                        self.logger.info(f"AI Player 2 chose action {action}, new state: {state}")

                    if gui is not None:
                        wins, losses, draws = self.ai_player.get_game_results()
                        gui.draw_board()
                        gui.update_score(wins, losses, draws)
                        
                        

                    if self.is_game_over() or self.is_draw():
                        break

                    state = next_state                    
                    self.current_player = 2 if self.current_player == 1 else 1  # Switch players
                                        
            
                if self.is_game_over() or self.is_draw():
                    if self.is_draw():
                        game_result = 'draw'
                        self.draws += 1 
                        self.logger.info(f"Game result: draw, AI1 wins: {self.ai_wins}, AI2 wins: {self.ai2_wins}, Draws: {self.draws}")
                        self.reset()  # Reset the game after it's over
                    else:
                        # The winner of this game is the current player
                        winner = self.current_player
                        # The loser of this game is the other player
                        self.loser = 2 if self.current_player == 1 else 1
                        # Set the starting player of the next game to be the loser of this game
                        self.starting_player = self.loser
                        # Update the win counts
                        if winner == 1:  # AI player won
                            self.ai_wins += 1
                            game_result = 'win'
                        # elif self.winner == 2:  # Assuming 2 represents the Human player
                        #     self.human_wins += 1
                        else:  # Human player won
                            self.ai2_wins += 1
                            game_result = 'loss'

                    self.ai_player.record_game_result(game_result)
                    # Log the result of the game and the updated win counts
                    self.logger.info(f"Game result: AI player {game_result}, AI wins: {self.ai_wins}, AI2 wins: {self.ai2_wins}")
                    self.reset()  # Reset the game after it's over
                else:
                    self.logger.warning("Game ended without a draw or a win.")

                self.logger.info(f"Total AI wins: {self.ai_wins}, Total AI2 wins: {self.ai2_wins}, Total Draws: {self.draws}")
                self.current_player = 2 if self.current_player == 1 else 1  # Switch players

            except Exception as e:
                self.logger.error(f"Exception occurred in episode {episode}, Game State: {state}, Last Action: {action}, Exception: {str(e)}")
                continue  # Skip to the next iteration instead of breaking the loop

            finally:
                # This block ensures that the game always processes events
                if gui:
                    gui._handle_events()

            # Train the AI player
            self.ai_player.train()
            self.ai_player2.train()
            self.ai_player.adjust_epsilon(episode)
            self.ai_player2.adjust_epsilon(episode)
            # Save the model every 100 episodes and after the final episode
            if episode % 100 == 0 or episode == episodes - 1:
                self.ai_player.save_best_model(model_path, episode)
                self.ai_player2.save_best_model(model_path + "_2", episode)
                self.ai_player.update_target_model()
                self.ai_player2.update_target_model()

                # After saving model, plot durations and record performance
                # Just for visualizig the performance of the AI
                # self.ai_player.plot_durations()
                # self.ai_player2.plot_durations()
                # self.ai_player.record_performance()
                # self.ai_player2.record_performance()
                # self.ai_player.plot_q_values()
                # self.ai_player2.plot_q_values()

            # Update end_time after each episode
            self.end_time = time.time()

        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Time taken for {episodes} episodes: {self.end_time - self.start_time} seconds")

        self.logger.info(f"Total AI wins: {self.ai_wins}, Total AI2 wins: {self.ai2_wins}, Total Draws: {self.draws}")
        # Run the GUI at the end
        if gui is not None:
            wins, losses, draws = self.ai_player.get_game_results()
            gui.draw_board()
            gui.update_score(wins, losses, draws)
