import os
import neat
import pyautogui
from PIL import Image
import pytesseract
import time
import visualize
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from checkpoint import Checkpointer
import pickle
import random

def split_image_into_grid(image, num_rows, num_cols):
    # Calculate the size of each grid cell
    img_width, img_height = image.size
    cell_width = img_width // num_cols
    cell_height = img_height // num_rows

    # List to hold the pieces
    pieces = []

    # Loop over the rows and columns
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * cell_width
            upper = row * cell_height
            right = left + cell_width
            lower = upper + cell_height

            # Define the bounding box and crop the image
            bbox = (left, upper, right, lower)
            crop_piece = image.crop(bbox)
            pieces.append(crop_piece)

    return pieces

def eval_genomes(genomes, config):
    neural_nets = []
    for genome_id, genome in genomes:
        # Initialize fitness, create net, store the neural nets
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        neural_nets.append(net)

    prev_score = 0
    moves_without_improvement = 0
    with open('winner.pkl', 'rb') as input_file:
      genome1 = pickle.load(input_file)
    net = neat.nn.FeedForwardNetwork.create(genome1, config)
    for index, net1 in enumerate(neural_nets):
      playing = True
      pyautogui.click(10, 660)
      while playing:
        # Take a screenshot of the screen
        screenshot = pyautogui.screenshot()
        # Store the tiles
        game_state = screenshot.crop((80, 700, 1050, 1680))
        game_status = screenshot.crop((250, 1050, 900, 1200))
        game = pytesseract.image_to_string(game_status)
        # If win / loss detected, job is done
        if 'game over' in game.lower() or 'you win' in game.lower():
          playing = False
          print('Win / loss!')
          break
        pieces = split_image_into_grid(game_state, 4, 4)

        # Use OCR to detect board state on screen
        board = []
        for piece in pieces:
          text = pytesseract.image_to_string(piece.crop((10, 30, 200, 300)), config="--psm 6 -c tessedit_char_whitelist=0123456789")
          only_digits = ''.join(filter(str.isdigit, text))
          if only_digits == '':
            text = pytesseract.image_to_string(piece.crop((10, 30, 200, 300)), config="--psm 13 -c tessedit_char_whitelist=0123456789")
            only_digits = ''.join(filter(str.isdigit, text))
          if only_digits != '' and int(only_digits) % 2 != 0:
            only_digits = ''
          board.append(0 if only_digits == '' else float(only_digits))
        # Determine neural net's best move based off of board state
        output = net.activate(board)
        neuron_values = output
        move = output.index(max(output))
        print('Board State:')
        print(board[1:4])
        print(board[4:8])
        print(board[8:12])
        print(board[12:16])
        # Make the move
        if move == 0:
          pyautogui.press('up')
        elif move == 1:
          pyautogui.press('down')
        elif move == 2:
          pyautogui.press('left')
        else:
          pyautogui.press('right')
        # Score is sum of the board
        score_num = sum(board)
        print(f'Score: {score_num}')
        print('')
        # Update the fitness
        newFitness = (0 if score_num == '' else float(score_num))
        genomes[index][1].fitness = newFitness
        # If it gets stuck, go next
        if newFitness == prev_score:
          moves_without_improvement += 1
        else:
          moves_without_improvement = 0
        prev_score = newFitness
        if moves_without_improvement > 1:
          choice = random.randint(0,3)
          if choice == 0:
            pyautogui.press('up')
          elif choice == 1:
            pyautogui.press('down')
          elif choice == 2:
            pyautogui.press('left')
          else:
            pyautogui.press('right')
      # Click to create a new game
      pyautogui.click(500, 300)
      time.sleep(1)
      pyautogui.click(450, 250)
      pyautogui.click(10, 660)
      moves_without_improvement = 0
      prev_score = 0


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = Checkpointer.restore_checkpoint('./neat-checkpoint-357730')
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(3))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)
    draw_net(config, winner, True)
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    run('./config-feedforward.txt')