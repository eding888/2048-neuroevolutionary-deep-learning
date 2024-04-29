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

import logic
import pickle

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot
def flatten_2d_array(array_2d):
    flattened_array = []
    for sublist in array_2d:
        flattened_array.extend(sublist)
    return flattened_array
yx = 0
def eval_genomes(genomes, config):
    global yx
    if yx > 100:
      pyautogui.click(10, 660)
      yx = 0
    yx += 1
    neural_nets = []
    for genome_id, genome in genomes:
        # Initialize genome initial fitness levels
        genome.fitness = 0
        # Initialize neural net from genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # Store those neural nets
        neural_nets.append(net)
    for index, net in enumerate(neural_nets):
      #Create game board
      mat = logic.start_game()
      prev_score = 0
      moves_without_improvement = 0
      while(True):
        #Utilize neural net to determine its desired move based on game state
        output = net.activate(flatten_2d_array(mat))
        move = output.index(max(output))

        # Make the move based off of game logic
        if(move == 0):
          mat, flag = logic.move_up(mat)
          status = logic.get_current_state(mat)
          if(status == 'GAME NOT OVER' and flag):
            logic.add_new_2(mat)
  
          elif(status != 'GAME NOT OVER'):
            genomes[index][1].fitness = genomes[index][1].fitness * 3
            break
    
        elif(move == 1):
          mat, flag = logic.move_down(mat)
          status = logic.get_current_state(mat)
          if(status == 'GAME NOT OVER' and flag):
            logic.add_new_2(mat)
          elif(status != 'GAME NOT OVER'):
            genomes[index][1].fitness = genomes[index][1].fitness * 3
            break

        elif(move == 2):
          mat, flag = logic.move_left(mat)
          status = logic.get_current_state(mat)
          if(status == 'GAME NOT OVER' and flag):
            logic.add_new_2(mat)
          elif(status != 'GAME NOT OVER'):
            genomes[index][1].fitness = genomes[index][1].fitness * 3
            break

        else:
          mat, flag = logic.move_right(mat)
          status = logic.get_current_state(mat)
          if(status == 'GAME NOT OVER' and flag):
            logic.add_new_2(mat)
          elif(status != 'GAME NOT OVER'):
            # We like when it wins/loses organically
            genomes[index][1].fitness = genomes[index][1].fitness * 3
            break

        # Score is the sum of all squares on the board
        score = sum(flatten_2d_array(mat))
        genomes[index][1].fitness = score

        # If the net gets stuck for a while, pass onto the next specimen
        if score == prev_score:
          moves_without_improvement += 1
        else:
          moves_without_improvement = 0
        prev_score = score
        if moves_without_improvement > 10:
          # We do not like it when it gets stuck, decentivise its weight
          genomes[index][1].fitness = genomes[index][1].fitness * 0.2
          break

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
    p.add_reporter(neat.Checkpointer(10000))

    # Run for many generations
    winner = p.run(eval_genomes, 100)
    with open('winner3.pkl', 'wb') as output:
      pickle.dump(winner, output, 1) 

    print('\nBest genome:\n{!s}'.format(winner))
    draw_net(config, winner, True)


if __name__ == '__main__':
    run('./config-feedforward.txt')