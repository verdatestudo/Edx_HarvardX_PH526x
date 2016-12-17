'''
Tic-Tac-Toe Datacamp
from HarvardX: PH526x Using Python for Research on edX

Last Updated: 2016-Dec-16
First Created: 2016-Dec-16
Python 3.5
Chris
'''

import numpy as np
import random
import time
import matplotlib.pyplot as plt

def create_board():
    '''
    Create blank board.
    '''
    return np.zeros((3, 3))

def place(board, player, position):
    '''
    Takes a numpy 3x3 array, int player (1 or 2) and a tuple position (x, y).
    '''
    if not board[position[0]][position[1]]:
        board[position[0]][position[1]] = player

def possibilities(board):
    '''
    Returns a list of tuples of empty board squares.
    '''
    npp = np.where(board == 0)
    return [(npp[0][x], npp[1][x]) for x in range(len(npp[0]))]

def random_place(board, player):
    '''
    Makes a random move for the specified player.
    '''
    place(board, player, random.choice(possibilities(board)))

def row_win(board, player):
    '''
    Checks if the player has won by rows.
    '''
    for row in board:
        if np.all(row == player):
            return True
    return False

def col_win(board, player):
    '''
    Checks if the player has won by cols.
    '''
    for col in range(np.shape(board)[1]):
        if np.all(board[:,col] == player):
            return True
    return False

def diag_win(board, player):
    '''
    Checks if the player has won by diags.
    '''
    if np.all(np.diagonal(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
        return True
    else:
        return False

def evaluate(board):
    '''
    Checks if the game is over.
    Returns 1 or 2 for a player victory, -1 for a draw and 0 if the game is still playable.
    '''
    for player in [1, 2]:
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            return player
    if not possibilities(board):
        return -1

    return 0

def play_game(board='', player=1):
    '''
    Play a game using a random strategy for both players.
    '''
    if type(board) != np.ndarray:
        board = create_board()
    while evaluate(board) == 0:
        random_place(board, player)
        player = (player - 1) or 2
    return evaluate(board)

def play_strategic_game():
    '''
    Play a game using a random strategy for both players, except player 1 moves first and goes for the centre.
    '''
    board = create_board()
    place(board, 1, (1, 1))
    return play_game(board, player=2)

def time_test(game_type, num_trials):
    '''
    Game type is a function and num_trials is an int.
    Does a time test and plots a histogram of results of game_type for num_trials.
    '''
    start_time = time.clock()
    results = [game_type() for _ in range(num_trials)]
    end_time = time.clock()

    plt.hist(results, bins = [-1.5, -0.5, 0.5, 1.5, 2.5])
    plt.show()
    return end_time - start_time

print(time_test(play_strategic_game, 1000))
print(time_test(play_game, 1000))
