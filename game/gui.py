# Save this as gui.py under the game directory
import pygame
import pygame_gui
from pygame_gui.elements import UIPanel, UIButton, UILabel
from pygame_gui import UIManager
from pygame.locals import *
from game.engine import Game
import sys

# Game constants
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 800
GRID_SIZE = 6
SQUARE_SIZE = WINDOW_WIDTH // GRID_SIZE

pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

class GUI:
    def __init__(self, game):

        self.game = game
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        self.manager = UIManager((WINDOW_WIDTH, WINDOW_HEIGHT), './game/theme.json')


        # define heights
        self.score_panel_height = 100
        self.nav_bar_height = 100

        # separate navigation panel
        self.nav_bar = UIPanel(pygame.Rect((0, self.score_panel_height), (WINDOW_WIDTH, self.nav_bar_height)), 
                               manager=self.manager, object_id='#nav_bar')

        self.font = pygame.font.SysFont('Arial', 40)
        # separate scoreboard surface
        self.score_surface = pygame.Surface((WINDOW_WIDTH, self.score_panel_height))

        self.human_play_active = False  # Flag to indicate if playing against human
        self.is_human_playing = False  # Flag to indicate if playing against human
        self.running = True  # Add this line to manage the running state

        self._create_UI_elements()  # Call this method to create UI elements

    def _create_UI_elements(self):
        button_width = 80
        button_height = 40  # Reduced height for visibility
        button_margin = 10

        # Calculate positions to ensure buttons are within the nav_bar
        x_position = button_margin
        y_position = (self.nav_bar_height - button_height) // 2  # Vertically center in the nav_bar    

        self.play_button = UIButton(
        relative_rect=pygame.Rect((x_position, y_position), (button_width, button_height)),
        text='Play',
        manager=self.manager,
        container=self.nav_bar,
        object_id="play_button"
        )
        x_position += button_width + button_margin

        self.pause_button = UIButton(
            relative_rect=pygame.Rect((x_position, y_position), (button_width, button_height)),
            text='Pause',
            manager=self.manager,
            container=self.nav_bar,
            object_id="pause_button"
        )
        x_position += button_width + button_margin

        self.quit_button = UIButton(
            relative_rect=pygame.Rect(((x_position, y_position), (button_width, button_height))),
            text='Quit',
            manager=self.manager,
            container=self.nav_bar,
            object_id="quit_button"  # Removed the hash (#) symbol
        )
        x_position += button_width + button_margin


        # Force a redraw of buttons (not typically necessary, but good for testing)
        self.play_button.rebuild()
        self.pause_button.rebuild()
        self.quit_button.rebuild()

    def draw_board(self):
        self._handle_events()
        self.window.fill(WHITE)

        # Draw game elements first
        self._draw_grid()
        self._draw_pieces()
        self._draw_score_board()

        # Force the manager to update - this processes any UI changes
        self.manager.update(0.05)  # Time delta is small, just to force the update

        # Draw UI elements (like buttons) after game elements
        self.manager.draw_ui(self.window)

        # Force the entire screen to update
        pygame.display.update()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.VIDEORESIZE:
                self.window = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.manager.set_window_resolution((event.w, event.h))
                self.nav_bar.kill()  # Remove the old nav bar

                # Recreate the nav bar with new window size
                self.nav_bar = UIPanel(pygame.Rect((0, self.score_panel_height), (event.w, self.nav_bar_height)), # start nav_bar at y=score_panel_height
                                    manager=self.manager,
                                    object_id='#nav_bar')
                self._create_UI_elements()

            elif event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.play_button:
                        print("Play Button Pressed")
                        self.game.resume_game()
                    elif event.ui_element == self.pause_button:
                        print("Pause Button Pressed")
                        self.game.pause_game()
                    elif event.ui_element == self.quit_button:
                        print("Quit Button Pressed")
                        self.running = False
                        pygame.quit()
                        sys.exit()
                    elif event.ui_element == self.play_again_button:
                        print("Play Again Button Pressed")
                        self.game.reset()  # Reset the game state
                        self.end_game_window.kill()  # Close the pop-up window
                        self.manager.clear_and_reset()  # Reset the UI managers
                        self.game.play_human(self)  # Restart the game
                    elif event.ui_element == self.pop_up_quit_button:  # Handle pop-up quit button
                        print("Pop-up Quit Button Pressed")
                        self.running = False
                        pygame.quit()
                        sys.exit()

            self.manager.process_events(event)



    def _draw_grid(self):
        window_width, window_height = self.window.get_size()
        navbar_height = self.nav_bar_height + self.score_panel_height  # navbar + score board height
        board_size = min(window_width, window_height - navbar_height)
        square_size = board_size / GRID_SIZE
        offset_x = (window_width - board_size) / 2
        offset_y = (window_height - board_size + navbar_height) / 2

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = int(col * square_size + offset_x)
                y = int(row * square_size + offset_y)
                pygame.draw.rect(self.window, WHITE, (x, y, int(square_size), int(square_size)))

        for i in range(GRID_SIZE+1):
            pygame.draw.line(self.window, (0,0,0), (int(offset_x), int(offset_y + i * square_size)), (int(offset_x + board_size), int(offset_y + i * square_size))) # horizontal grid lines
            pygame.draw.line(self.window, (0,0,0), (int(offset_x + i * square_size), int(offset_y)), (int(offset_x + i * square_size), int(offset_y + board_size))) # vertical grid lines

    def _draw_pieces(self):
        window_width, window_height = self.window.get_size()
        navbar_height = self.nav_bar_height + self.score_panel_height  # navbar + score board height
        board_size = min(window_width, window_height - navbar_height)
        square_size = board_size / GRID_SIZE
        offset_x = (window_width - board_size) / 2
        offset_y = (window_height - board_size + navbar_height) / 2

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = int(col * square_size + offset_x)
                y = int(row * square_size + offset_y)
                if self.game.board[row][col] == 1:
                    pygame.draw.circle(self.window, RED, (x + square_size // 2, y + square_size // 2), square_size // 2 - 5)
                elif self.game.board[row][col] == 2:
                    pygame.draw.circle(self.window, YELLOW, (x + square_size // 2, y + square_size // 2), square_size // 2 - 5)

    def run(self, episodes=1000, batch_size=50):
        while self.running:
            time_delta = pygame.time.Clock().tick(60) / 1000.0
            self._handle_events()

            if self.is_human_playing:
                selected_move = self.get_player_input()
                if selected_move is not None and selected_move in self.game.get_valid_moves():
                    self.game.make_move(selected_move, self.game.current_player)
                    self.game.current_player = 2 if self.game.current_player == 1 else 1
                    self.draw_board()
                    if self.game.is_game_over() or self.game.is_draw():
                        self._display_game_over_message()

            self.manager.update(time_delta)  # update the manager
            self.window.fill(WHITE)
            self._draw_grid()
            self._draw_pieces()

            self.manager.draw_ui(self.window)  # draw the GUI
            pygame.display.update()

        pygame.quit()

    def get_player_input(self):
        window_width, window_height = self.window.get_size()
        navbar_height = self.nav_bar_height + self.score_panel_height  # navbar + score board height
        board_size = min(window_width, window_height - navbar_height)
        square_size = board_size / GRID_SIZE
        offset_x = (window_width - board_size) / 2
        offset_y = navbar_height  # Start of the board in y direction
        board_end_y = offset_y + board_size  # End of the board in y direction

        valid_moves = self.game.get_valid_moves()
        selected_move = None
        while selected_move not in valid_moves:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    if offset_x <= x < offset_x + board_size and offset_y <= y < board_end_y:  # Check if the click is within the board
                        selected_move = int((x - offset_x) // square_size)
                        return selected_move if selected_move in valid_moves else None
                # Process other events like quitting or button presses
                self._handle_events()
        return None  # If no valid move was selected
    
    def _display_game_over_message(self):
        message = "Game Over! You " + ("won!" if self.game.winner == 2 else "lost!")
        message += "\nPlay again?"

        # Ensure the pop-up window is not created off-screen
        pop_up_rect = pygame.Rect((WINDOW_WIDTH // 4, WINDOW_HEIGHT // 4), (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        if pop_up_rect.x < 0 or pop_up_rect.y < 0:
            pop_up_rect.x, pop_up_rect.y = 100, 100  # Default position

        # Create a pop-up window for the end game message
        self.end_game_window = pygame_gui.elements.UIWindow(
            rect=pop_up_rect,
            manager=self.manager,
            window_display_title='Game Over',
            object_id="end_game_window"
        )

        # Show the window
        self.end_game_window.show()

        # Create Play Again and Quit buttons
        self.play_again_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 70), (100, 30)),
            text='Play Again',
            manager=self.manager,
            container=self.end_game_window,
            object_id="play_again_button"
        )
        self.pop_up_quit_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((120, 70), (100, 30)),
            text='Quit',
            manager=self.manager,
            container=self.end_game_window,
            object_id="pop_up_quit_button"
        )

        # Display the end game message
        self.end_game_message_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 10), (self.end_game_window.rect.width - 20, 50)),
            text=message,
            manager=self.manager,
            container=self.end_game_window
        )

        # Show the window and handle its events
        self.end_game_window.show()
        self.manager.update(0.05)  # Force update
        self.manager.draw_ui(self.window)
        pygame.display.update()

    def _draw_score_board(self):
        
        self.window.blit(self.score_surface, (0, 0))

    
    def update_score(self, wins, losses, draws):
        if self.is_human_playing:
            score_text = self.font.render(f'AI Wins: {self.game.ai_wins}, Human Wins: {self.game.human_wins}', True, (0, 0, 0))
            score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 50))  # Center align the score text
            self.score_surface.fill((100, 100, 100))  # Fill the surface with white before adding text
            self.score_surface.blit(score_text, score_rect)
        else:
            score_text = self.font.render(f'AI Wins: {self.game.ai_wins}, AI2 Wins: {self.game.ai2_wins}', True, (0, 0, 0))
            score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 50))  # Center align the score text
            self.score_surface.fill((100, 100, 100))  # Fill the surface with white before adding text
            self.score_surface.blit(score_text, score_rect)
