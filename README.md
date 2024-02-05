Connect Four AI Game
Description
This project implements a Connect Four game with a sophisticated AI opponent. The AI uses a Dueling Convolutional Neural Network (CNN) and Prioritized Experience Replay for decision-making. The game features a graphical user interface (GUI) built using Pygame, providing an interactive and user-friendly experience.

Features
AI Opponent: Advanced AI using Dueling CNN and reinforcement learning.
Interactive GUI: Built with Pygame for a responsive and intuitive gameplay experience.
Customizable Settings: Adjustable grid size, game parameters, and AI settings.
Logging: Comprehensive logging for tracking game progress and debugging.

Installation
To run this game, you need Python and the necessary libraries installed on your system.

Prerequisites
Python 3.x
Pygame
PyTorch
Numpy

Setup
Clone the repository:
'''
git clone [repository URL]
'''
Install dependencies:
'''
pip install -r requirements.txt
'''
Usage
To start the game, navigate to the project directory and run:
'''
python main.py
'''

Command-Line Arguments
'''
--grid_size: Set the size of the game grid (default: 6).
--episodes: Number of episodes for AI training (default: 5000).
--batch_size: Batch size for AI training (default: 32).
--model_path: Path to save or load the AI model.
--human_play: Enable human vs. AI mode.
'''

Example:
'''
python main.py --grid_size 6 --episodes 1000 --human_play
'''

AI and Training
The AI uses a Dueling CNN model, learning through reinforcement learning techniques. Training involves playing numerous games against itself, gradually improving its strategy.

Model Architecture
Input Layer: Adapted to the grid size.
Hidden Layers: Convolutional layers with residual connections.
Output Layer: Separate streams for value and advantage estimation.

GUI
The GUI is built using Pygame and provides a simple and intuitive interface for playing the game. It includes visual representations of the game board, interactive buttons, and real-time score updates.

Logs and Performance
Logging is used extensively for debugging and performance tracking. Logs include game results, AI decisions, and error messages.

Contributing
Contributions to the project are welcome. Please follow the standard procedures for contributing to open-source projects.

Fork the repository.
Create a new branch for your feature.
Commit your changes.
Push to the branch.
Open a pull request.
License
[Specify License]


