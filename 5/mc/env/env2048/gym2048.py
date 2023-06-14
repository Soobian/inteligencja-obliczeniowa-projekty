import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import mc.env.env2048.colors


class Gym2048(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        self.board = []
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2048, shape=(4, 4), dtype=np.uint16)
        self.reset()
        self._window = None
        self._clock = None

    def reset(self):
        self.board = np.zeros((4, 4), dtype=np.uint16)
        self._add_random_tile()
        self._add_random_tile()

        return self.board

    def step(self, action):
        prev_board = self.board.copy()

        if action == 0:  # Move up
            self._move_up()
        elif action == 1:  # Move down
            self._move_down()
        elif action == 2:  # Move left
            self._move_left()
        elif action == 3:  # Move right
            self._move_right()

        done = self._is_game_over()

        if not done and np.any(prev_board != self.board):
            self._add_random_tile()

        reward = self._calculate_reward()

        return self.board.copy(), reward, done, {}

    def render(self, mode="human"):
        assert mode
        if mode == "human":
            board = self.board
            for i, row in enumerate(board):
                print(" | ".join("{:4d}".format(tile) if tile != 0 else "    " for tile in row))
                if i < len(board) - 1:
                    print("-" * (7 * len(board) - 1))
            print("\n")
        elif mode == "rgb_array":
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

    def close(self):
        # Clean up any resources or connections used by the environment
        pass

    def _move_up(self):
        self.board = np.rot90(self.board)
        self._move_left()
        self.board = np.rot90(self.board, k=-1)

    def _move_down(self):
        self.board = np.rot90(self.board, k=-1)
        self._move_left()
        self.board = np.rot90(self.board)

    def _move_left(self):
        for row in range(4):
            merged = [False] * 4
            for col in range(1, 4):
                if self.board[row][col] != 0:
                    current_col = col
                    while current_col > 0 and (
                            self.board[row][current_col - 1] == 0 or self.board[row][current_col - 1] ==
                            self.board[row][col]):
                        if self.board[row][current_col - 1] == self.board[row][col] and not merged[current_col - 1]:
                            self.board[row][current_col - 1] += self.board[row][col]
                            self.board[row][col] = 0
                            merged[current_col - 1] = True
                        else:
                            self.board[row][current_col - 1] = self.board[row][col]
                            self.board[row][col] = 0
                        current_col -= 1

    def _move_right(self):
        self.board = np.rot90(self.board, k=2)
        self._move_left()
        self.board = np.rot90(self.board, k=-2)

    def _add_random_tile(self):
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) > 0:
            random_cell = empty_cells[np.random.randint(len(empty_cells))]
            self.board[random_cell[0]][random_cell[1]] = np.random.choice([2, 4])

    def _is_game_over(self):
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) > 0:
            return False

        for row in range(4):
            for col in range(4):
                cell_value = self.board[row][col]
                if (row > 0 and self.board[row - 1][col] == cell_value) or \
                   (row < 3 and self.board[row + 1][col] == cell_value) or \
                   (col > 0 and self.board[row][col - 1] == cell_value) or \
                   (col < 3 and self.board[row][col + 1] == cell_value):
                    return False

        return True

    def _calculate_reward(self):
        #reward = np.sum(self.board)
        reward = np.amax(self.board)
        return reward


