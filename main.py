try:
    import numpy as np
except ImportError:
    raise

from easyAI import TwoPlayerGame, AI_Player, Negamax


class ClumsyConnectFour(TwoPlayerGame):
    def __init__(self, players, board=None):
        self.players = players
        self.board = (
            board if (board is not None) else (np.array([[0 for i in range(7)] for j in range(6)]))
        )
        self.current_player = 1

    def possible_moves(self):
        return [i for i in range(7) if (self.board[:, i].min() == 0)]

    def make_move(self, column):
        rand_move = np.random.choice(np.arange(-1, 2), p=[0.05, 0.9, 0.05])
        column = min(column + rand_move, 6)
        column = max(column, 0)
        line = np.argmin(self.board[:, column] != 0)
        self.board[line, column] = self.current_player

    def show(self):
        print(
            "\n"
            + "\n".join(
                ["0 1 2 3 4 5 6", 13 * "-"]
                + [
                    " ".join([[".", "O", "X"][self.board[5 - j][i]] for i in range(7)])
                    for j in range(6)
                ]
            )
        )

    def lose(self):
        return find_four(self.board, self.opponent_index)

    def is_over(self):
        return (self.board.min() > 0) or self.lose()

    def scoring(self):
        return -100 if self.lose() else 0

def find_four(board, current_player):
    for pos, direction in POS_DIR:
        streak = 0
        while (0 <= pos[0] <= 5) and (0 <= pos[1] <= 6):
            if board[pos[0], pos[1]] == current_player:
                streak += 1
                if streak == 4:
                    return True
            else:
                streak = 0
            pos = pos + direction
    return False

POS_DIR = np.array(
    [[[i, 0], [0, 1]] for i in range(6)]
    + [[[0, i], [1, 0]] for i in range(7)]
    + [[[i, 0], [1, 1]] for i in range(1, 3)]
    + [[[0, i], [1, 1]] for i in range(4)]
    + [[[i, 6], [1, -1]] for i in range(1, 3)]
    + [[[0, i], [1, -1]] for i in range(3, 7)]
)


class Tournament():
    def __init__(self, tournament_length):
        self.ranking = [0, 0]
        self.tournament_length = tournament_length
        self.player1 = None
        self.player2 = None

    def define_players(self, level1, level2):
        self.player1 = Negamax(level1)
        self.player2 = Negamax(level2)

    def start_round(self, switch_player):
        game = ClumsyConnectFour([AI_Player(self.player1), AI_Player(self.player2)])
        if switch_player:
            game.switch_player()
        game.play()
        if game.lose():
            self.ranking[game.opponent_index - 1] += 1
        else:
            print("DRAW")

    def start_tournament(self):
        self.define_players(3, 2)
        for _ in range(self.tournament_length):
            switch_player = np.random.choice(np.arange(0, 2), p=[0.5, 0.5])
            self.start_round(switch_player)


if __name__ == "__main__":
    tournament = Tournament(100)
    tournament.start_tournament()
    print(tournament.ranking)

