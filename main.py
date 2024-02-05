# main.py
from game.gui import GUI
from game.engine import Game
from ai.model import AIPlayer
import logging
import time
import traceback
import argparse
import os

def main():
    # Set up logging
    logging.basicConfig(filename='game.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="A Connect Four AI game implemented with Deep Q-learning.")
    parser.add_argument('--grid_size', type=int, default=6, help="Size of the game grid as [rows, columns]")
    parser.add_argument('--episodes', type=int, default=5000, help="Number of episodes for training the AI")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the AI training")
    parser.add_argument('--model_path', type=str, default='../connect-four-ai/ai_model.pth', help="Path to save/load the AI model")
    parser.add_argument('--human_play', action='store_true', help="Enable human vs AI mode")

    # Parse the command line arguments
    args = parser.parse_args()

    try:
        # Initialize the game and GUI
        game = Game(grid_size=args.grid_size)
        gui = GUI(game)

        # Record start time
        start_time = time.time()

        # Check if human play mode is activated
        if args.human_play:
            logger.info("Human vs AI mode activated.")
            ai_model_path = args.model_path if os.path.exists(args.model_path) else None
            logger.info(f"AI model path: {ai_model_path}")
            game.toggle_human_play(gui, model_path=ai_model_path)
            game.play_human(gui=gui)
        else:
            # AI training mode
            logger.info("AI training mode activated.")
            game.play(episodes=args.episodes, batch_size=args.batch_size, model_path=args.model_path, gui=gui)

        # Record end time and calculate duration
        end_time = time.time()
        logger.info(f"Total run time: {end_time - start_time} seconds")

        # Run the GUI
        gui.run()

    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
