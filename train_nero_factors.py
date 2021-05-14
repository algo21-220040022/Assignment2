from fitness_calculation import fitness_cal
import neat
import pickle
from parameters import GENERATION_NUM, FACTOR_CATEGORY
from data_process import Data_Process
from itertools import count
from configparser import ConfigParser
import os
from copy import deepcopy
import numpy as np
from multiprocessing import Pool

data_handler = Data_Process()

def evalGenome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = fitness_cal(net, data_handler)
    return fitness

def create_pop(last_factor_winner, num_genomes):
    genome_indexer = count(1)
    new_genomes = {}
    for i in range(num_genomes):
        key = next(genome_indexer)
        tmp_genome = deepcopy(last_factor_winner)
        tmp_genome.key = key
        new_genomes[key] = tmp_genome
    return new_genomes

def my_sigmoid(x):
    return 1/(1+np.exp(-0.3*x))

def my_activation(x):
    return (x)/(10e-10+10*abs(x/30)**(1/3))

def run(config_file, last_factor_winner):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    config.genome_config.add_activation('my_activation', my_activation)
    config.genome_config.add_activation('my_sigmoid', my_sigmoid)
    # Create the population, which is the top-level object for a NEAT run.
    if last_factor_winner != None:
        print(f"Load the last best factor network!")
        p = neat.Population(config, initial_state=(0,0,0))
        population = create_pop(last_factor_winner, 10)
        species = config.species_set_type(config.species_set_config, p.reporters)
        generation = 0
        species.speciate(config, population, generation)
        p.population = population
        p.species = species
        p.generation = generation
    else:
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 1000 generations.
    # pe = neat.ThreadedEvaluator(multiprocessing.cpu_count(), evalGenome) #Threading
    pe = neat.ThreadedEvaluator(20, evalGenome)
    # pe = neat.ParallelEvaluator(4, evalGenome)
    winner = p.run(pe.evaluate, GENERATION_NUM)
    pe.stop() # Threading
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    return winner_net, winner

def change_config_input_num(input_num,_config_file):
    config_parser = ConfigParser()
    config_parser.read(_config_file)
    config_parser.set("DefaultGenome", "num_inputs", str(input_num))
    if _config_file.split("-")[-1] == "volatility":
        config_parser.set("DefaultGenome", "activation_default", "my_sigmoid")
        config_parser.set("DefaultGenome", "activation_options", "my_sigmoid")
    else:
        config_parser.set("DefaultGenome", "activation_default", "my_activation")
        config_parser.set("DefaultGenome", "activation_options", "my_activation")
    config_parser.set("DefaultGenome", "bias_max_value", "0.1")
    config_parser.set("DefaultGenome", "bias_min_value", "-0.1")
    config_parser.set("DefaultGenome", "response_max_value", "2")
    config_parser.set("DefaultGenome", "response_min_value", "0.5")
    config_parser.set("DefaultGenome", "weight_max_value", "10")
    config_parser.set("DefaultGenome", "weight_min_value", "-10")
    config_parser.set("DefaultGenome", "weight_init_mean", "0")
    config_parser.set("DefaultGenome", "weight_init_stdev", "1")
    with open(_config_file, "w+") as f:
        config_parser.write(f)



def main():
    config_file = "./config/config-rnn"
    train_data_pkls = os.listdir("./data_for_training/")
    train_data_pkls.sort(key=lambda x:int(x[:-4]))
    output_pkls = os.listdir("./nero_factors_model/")
    # factor_class = ["momentum", "growth"]
    factor_class = list(FACTOR_CATEGORY.keys())
    use_last_exist_winner = False

    for ky in factor_class:
        factor_list = FACTOR_CATEGORY[ky]
        change_config_input_num(len(factor_list),config_file)
        data_handler.set_factor_list(factor_list)
        last_factor_winner = None
        for pkl in train_data_pkls:
            if f"{ky}_{pkl[:8]}_winner.pkl" in output_pkls:
                print(f"{ky}_{pkl[:8]}.pkl has been run!" )
                if use_last_exist_winner:
                    last_factor_winner = pickle.load(open(f"./nero_factors_model/{ky}_{pkl[:8]}_winner.pkl", "rb"))
                continue
            print(f"Start training nero factor [{ky}] for {pkl[:8]}....")
            data_handler.load_training_data("./data_for_training/"+pkl)
            winner_net, winner = run(config_file, last_factor_winner)
            file = open(f"./nero_factors_model/{ky}_{pkl[:8]}.pkl", "wb")
            pickle.dump(winner_net, file)
            file.close()
            file = open(f"./nero_factors_model/{ky}_{pkl[:8]}_winner.pkl", "wb")
            pickle.dump(winner, file)
            file.close()
            last_factor_winner = winner
            print(f"Finish training nero factor [{ky}] for {pkl[:8]}!")

"""MultiProcessing"""
def multi_main(factor):
    config_file = f"./config/config-rnn-{factor}"
    train_data_pkls = os.listdir("./data_for_training/")
    train_data_pkls.sort(key=lambda x: int(x[:-4]))
    output_pkls = os.listdir("./nero_factors_model/")
    use_last_exist_winner = True
    ky = factor
    factor_list = FACTOR_CATEGORY[ky]
    change_config_input_num(len(factor_list),config_file)
    data_handler.set_factor_list(factor_list)
    last_factor_winner = None
    for pkl in train_data_pkls:
        if f"{ky}_{pkl[:8]}_winner.pkl" in output_pkls:
            print(f"{ky}_{pkl[:8]}.pkl has been run!")
            if use_last_exist_winner:
                last_factor_winner = pickle.load(open(f"./nero_factors_model/{ky}_{pkl[:8]}_winner.pkl", "rb"))
            continue
        print(f"Start training nero factor [{ky}] for {pkl[:8]}....")
        data_handler.load_training_data("./data_for_training/" + pkl)
        winner_net, winner = run(config_file, last_factor_winner)
        file = open(f"./nero_factors_model/{ky}_{pkl[:8]}.pkl", "wb")
        pickle.dump(winner_net, file)
        file.close()
        file = open(f"./nero_factors_model/{ky}_{pkl[:8]}_winner.pkl", "wb")
        pickle.dump(winner, file)
        file.close()
        last_factor_winner = winner
        print(f"Finish training nero factor [{ky}] for {pkl[:8]}!")

def single_factor_multi_main(train_data_pkls):
    factor = "composite_factor"

    config_file = f"./config/config-rnn-{factor}"
    output_pkls = os.listdir("./nero_factors_model/")
    use_last_exist_winner = True
    factor_list = FACTOR_CATEGORY[factor]
    change_config_input_num(len(factor_list),config_file)
    data_handler.set_factor_list(factor_list)
    last_factor_winner = None
    for pkl in train_data_pkls:
        if f"{factor}_{pkl[:8]}_winner.pkl" in output_pkls:
            print(f"{factor}_{pkl[:8]}.pkl has been run!")
            if use_last_exist_winner:
                last_factor_winner = pickle.load(open(f"./nero_factors_model/{factor}_{pkl[:8]}_winner.pkl", "rb"))
            continue
        print(f"Start training nero factor [{factor}] for {pkl[:8]}....")
        data_handler.load_training_data("./data_for_training/" + pkl)
        winner_net, winner = run(config_file, last_factor_winner)
        file = open(f"./nero_factors_model/{factor}_{pkl[:8]}.pkl", "wb")
        pickle.dump(winner_net, file)
        file.close()
        file = open(f"./nero_factors_model/{factor}_{pkl[:8]}_winner.pkl", "wb")
        pickle.dump(winner, file)
        file.close()
        last_factor_winner = winner
        print(f"Finish training nero factor [{factor}] for {pkl[:8]}!")

def split_list_to_n_parts(_list, n):
    for i in range(0, len(_list), int(np.ceil(len(_list)/n))):
        yield _list[i:i+int(np.ceil(len(_list)/n))]

if __name__ == '__main__':
    factor_class = list(FACTOR_CATEGORY.keys())
    pool = Pool(len(factor_class))
    pool.map(multi_main, factor_class)
    pool.close()
    pool.join()
    # all_train_data_pkls = os.listdir("./data_for_training/")
    # all_train_data_pkls.sort(key=lambda x: int(x[:-4]))
    # all_train_data_pkls =  ['20200228.pkl', '20200331.pkl', '20200430.pkl', '20200529.pkl']
    # train_data_pkls_group = list(split_list_to_n_parts(all_train_data_pkls, 8))
    # pool = Pool(5)
    # pool.map(single_factor_multi_main, train_data_pkls_group)
    # pool.close()
    # pool.join()