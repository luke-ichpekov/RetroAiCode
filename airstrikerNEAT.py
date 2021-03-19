import retro
import numpy as np
import cv2
import neat
import pickle


env = retro.make('Airstriker-Genesis')

imgarray = []



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        ob = env.reset()
        action = env.action_space.sample()
        inx, iny, inc, = env.observation_space.shape
        inx= int(inx/8)
        iny= int(iny/ 8)
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        fitness_current= 0
        current_frame =0
        initialScore = 0
        initialLives = 3
        done = False

        while not done:
            env.render()
            current_frame+=0.1
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            imgarray = np.ndarray.flatten(ob)
            nnOutput = net.activate(imgarray)
            ob, rew, done, info = env.step(nnOutput)
            # print(nnOutput)
            score = info['score']
            lives = info['lives']

            if(score > initialScore):
                fitness_current += 5000
                initialScore=score
            if(lives < initialLives):
                fitness_current -=2000
                initialLives = lives
           # fitness_current += current_frame
            print(genome_id, fitness_current)
            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward.txt')


p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

