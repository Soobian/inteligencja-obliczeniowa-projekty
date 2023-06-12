import gymnasium as gym
import numpy as np

from gymnasium import spaces
from .game import Game2048
import pygame
import env2048.envs.color_constants as colors


class Env2048(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
            self, render_mode=None, board_size=4, binary=True, extractor="cnn", invalid_move_warmup=16,
            invalid_move_threshold=0.1,
            penalty=-512):
        """
        Iniatialize the environment.


        Parameters
        ----------
        board_size : int
            Size of the board. Default=4
        binary : bool
            Use binary representation of the board(power 2 matrix). Default=True
        extractor : str
            Type of model to extract the features. Default=cnn
        invalid_move_warmup : int
            Minimum of invalid movements to finish the game. Default=16
        invalid_move_threshold : float
                    How much(fraction) invalid movements is necessary according to the total of moviments already executed. to finish the episode after invalid_move_warmup. Default 0.1
        penalty : int
            Penalization score of invalid movements to sum up in reward function. Default=-512
        seed :  int
            Seed
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.__render_mode = render_mode
        self._window = None
        self._clock = None

        self.state = np.zeros(board_size * board_size)
        self.__binary = binary
        self.__extractor = extractor

        if self.__binary is True:
            if extractor == "cnn":
                self.observation_space = spaces.Box(
                    0, 1, (board_size, board_size, 16 + (board_size - 4)), dtype=np.uint32
                )
            else:
                self.observation_space = spaces.Box(
                    0, 1, (board_size * board_size * (16 + (board_size - 4)),), dtype=np.uint32
                )
        else:
            if extractor == "mlp":
                self.observation_space = spaces.Box(0, 2 ** 16, (board_size * board_size,), dtype=np.uint32)
            else:
                ValueError("Extractor must to be mlp when observation space is not binary")

        self.action_space = spaces.Discrete(4)  # Up, down, right, left

        if penalty > 0:
            raise ValueError("The value of penalty needs to be between [0, -inf)")
        self.__game = Game2048(board_size, invalid_move_warmup, invalid_move_threshold, penalty)
        self.__n_iter = 0
        self.__done = False
        self.__total_score = 0
        self.__board_size = board_size

    def step(self, action):
        """
        Execute an action.

        Parameters
        ----------
        action : int
            Action selected by the model.
        """
        reward = 0
        info = dict()

        before_move = self.__game.get_board().copy()
        self.__game.make_move(action)
        self.__game.confirm_move()

        if self.__binary is True:
            self.__game.transform_board_to_power_2_mat()

            if self.__extractor == "cnn":
                self.state = self.__game.get_power_2_mat()
            else:
                self.state = self.__game.get_power_2_mat().flatten()
        else:
            self.state = self.__game.get_board().flatten()

        self.__done, penalty = self.__game.verify_game_state()
        reward = self.__game.get_move_score() + penalty
        self.__n_iter = self.__n_iter + 1
        after_move = self.__game.get_board()

        info["total_score"] = self.__game.get_total_score()
        info["steps_duration"] = self.__n_iter
        info["before_move"] = before_move
        info["after_move"] = after_move

        return self.state, reward, self.__done, self.__create_info()

    def __create_info(self):
        info = dict()
        info["total_score"] = self.__game.get_total_score()
        info["steps_duration"] = self.__n_iter
        info["before_move"] = self.__game.get_board()
        info["after_move"] = self.__game.get_board()
        return info

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.__n_iter = 0
        self.__done = False
        self.__total_score = 0
        self.__game.reset()

        if self.__binary is True:
            self.__game.transform_board_to_power_2_mat()
            if self.__extractor == "cnn":
                self.state = self.__game.get_power_2_mat()
            else:
                self.state = self.__game.get_power_2_mat().flatten()
        else:
            self.state = self.__game.get_board().flatten()
        return self.state, self.__create_info()

    def render(self, mode="human"):
        if self.__render_mode == "human":
            board = self.__game.get_board()
            for i, row in enumerate(board):
                print(" | ".join("{:4d}".format(tile) if tile != 0 else "    " for tile in row))
                if i < len(board) - 1:
                    print("-" * (7 * len(board) - 1))
            print("\n")
        elif self.__render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((400, 400))
        if self._clock is None:
            self._clock = pygame.time.Clock()

        # Clear the window
        self._window.fill((255, 255, 255))

        # Draw the game board
        board = self.__game.get_board()
        tile_size = 80
        margin = 10
        for i in range(len(board)):
            for j in range(len(board[i])):
                tile_value = board[i][j]
                tile_color = colors.EMPTY_TILE_COLOR  # Default color for empty tiles
                if tile_value != 0:
                    # Assign different colors based on tile values
                    tile_color = colors.TILES_COLOR[tile_value]
                    # Add more color assignments for higher tile values if needed

                # Calculate the position of the tile on the window
                x = j * (tile_size + margin)
                y = i * (tile_size + margin)

                # Draw the tile
                pygame.draw.rect(self._window, tile_color, (x, y, tile_size, tile_size))
                if tile_value != 0:
                    # Draw the tile value
                    font = pygame.font.Font(None, 36)
                    text = font.render(str(tile_value), True, (0, 0, 0))
                    text_rect = text.get_rect(center=(x + tile_size // 2, y + tile_size // 2))
                    self._window.blit(text, text_rect)

        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self._clock.tick(self.metadata["render_fps"])

    def get_board(self):
        """Get the board."""

        return self.__game.get_board()
