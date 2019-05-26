#!/usr/bin/env python

"""
__author__ = "Lech Szymanski and Vivian Breda"
__copyright__ = "Copyright 2019, COSC343"
__license__ = "GPL"
__version__ = "2.0.1"
__maintainer__ = "Lech Szymanski"
__email__ = "lechszym@cs.otago.ac.nz"
"""

from cosc343world import Creature, World
import numpy as np
import time
import matplotlib.pyplot as plt

# You can change this number to specify how many generations creatures are going to evolve over...
numGenerations = 300

# You can change this number to specify how many turns in simulation of the world for given generation
numTurns = 100

# You can change this number to change the world type.  You have two choices - world 1 or 2 (described in
# the assignment 2 pdf document)
worldType = 2

# You can change this number to change the world size
gridSize = 50

# You can set this mode to True to have same initial conditions for each simulation in each generation.  Good
# for development, when you want to have some determinism in how the world runs from generation to generation.
repeatableMode = True

generationFitness = np.zeros(shape=numGenerations)
generationAvgLifeTime = np.zeros(shape=numGenerations)
pos = 0
energy = 100


# This is a class implementing you creature a.k.a MyCreature.  It extends the basic Creature, which provides the
# basic functionality of the creature for the world simulation.  Your job is to implement the AgentFunction
# that controls creature's behaviour by producing actions in response to percepts.
class MyCreature(Creature):
    bonus_eater = 0

    # Initialisation function.  This is where you creature
    # should be initialised with a chromosome in random state.  You need to decide the format of your
    # chromosome and the model that it's going to give rise to
    #
    # Input: numPercepts - the size of percepts list that creature will receive in each turn
    #        numActions - the size of actions list that creature must create on each turn
    def __init__(self, numPercepts, numActions):

        # Set up the creature's chromosome to be a 9 (percepts) x 11 (possible actions) matrix,
        # initialized to a random state.
        self.chromosome = np.random.uniform(low=0, high=1, size=(numPercepts, numActions))
        self.fitness = 0

        # Do not remove this line at the end.  It calls constructors
        # of the parent classes.
        Creature.__init__(self)

    # This is the implementation of the agent function that is called on every turn, giving the
    # creature a chance to perform an action. The actions are chosen by the highest value in the
    # output array, which is created by multiplying the percepts array by the chromosome matrix.
    #
    # Input: percepts - a list of percepts
    #        numAction - the size of the actions list that needs to be returned
    def AgentFunction(self, percepts, numActions):
        global energy

        current_energy = self.getEnergy()

        percepts = (percepts + 1) / 4
        actions = np.dot(a=percepts, b=self.chromosome)

        if self.getEnergy() > energy:
            self.bonus_eater += 25

        energy = current_energy
        return actions.tolist()


def fitness(individual):
    bonus_effort = 0
    bonus_runner = 50

    if individual.isDead():
        if individual.timeOfDeath() > numTurns / 2:
            if individual.getEnergy() >= 50:
                bonus_effort = individual.bonus_eater + bonus_runner
        indv_fitness = individual.timeOfDeath() + individual.getEnergy() + bonus_effort
    else:
        if individual.getEnergy() >= 50:
            bonus_effort = individual.bonus_eater + (3 * bonus_runner) + 200
        else:
            bonus_effort = (2 * bonus_runner) + 100

        indv_fitness = numTurns + individual.getEnergy() + bonus_effort + 300

    return indv_fitness

    # The crossover function creates children from two random selected parents.
    # It adds the the parts by rows of the parents' chromosomes.

    # Input: parents - an array containing two selected creatures
def crossover(parents):
    child_1 = MyCreature(numCreaturePercepts, numCreatureActions)
    child_2 = MyCreature(numCreaturePercepts, numCreatureActions)

    child_1.chromosome = np.copy(parents[0].chromosome)
    child_2.chromosome = np.copy(parents[1].chromosome)

    crossover_point = np.random.randint(low=1, high=len(parents[0].chromosome[0]) - 1)

    child_1.chromosome[:, crossover_point:] = parents[1].chromosome[:, crossover_point:]
    child_2.chromosome[:, crossover_point:] = parents[0].chromosome[:, crossover_point:]

    return child_1, child_2


def mutate(individual, mutate_prob):
    row = np.random.randint(low=0, high=8)
    column = np.random.randint(low=0, high=10)

    # for i in range(len(individual.chromosome)):
    mutation_chance = np.random.uniform(low=0, high=1)

    if mutation_chance <= mutate_prob:
        individual.chromosome[row][column] = np.random.uniform(low=0, high=1)

    return individual


def population_fitness(all_creatures):
    total_fitness = 0
    for creature in all_creatures:
        total_fitness += creature.fitness

    return total_fitness


# Tournament Selection - choose the fittest pair out of 5 random individuals
def select_parents(all_creatures):
    indexes = np.random.randint(low=0, high=len(all_creatures) - 1, size=5)
    parents = []

    for index in indexes:
        parents.append(all_creatures[index])

    for p in range(3):
        parents.remove(min(parents, key=lambda creature: creature.fitness))
        p += 1
    return parents


# Pick 10% of the population elite
def pick_elite(all_creatures):
    sorted_population = list(sorted(all_creatures, key=lambda creature: creature.fitness, reverse=True))
    index = int(len(all_creatures) * 0.10)
    return sorted_population[:index]


# This function is called after every simulation, passing a list of the old population of creatures, whose fitness
# you need to evaluate and whose chromosomes you can use to create new creatures.
#
# Input: old_population - list of objects of MyCreature type that participated in the last simulation.  You
#                         can query the state of the creatures by using some built-in methods as well as any methods
#                         you decide to add to MyCreature class.  The length of the list is the size of
#                         the population.  You need to generate a new population of the same size.  Creatures from
#                         old population can be used in the new population - simulation will reset them to starting
#                         state.
#
# Returns: a list of MyCreature objects of the same length as the old_population.
def newPopulation(old_population):
    global numTurns
    global generationFitness
    global generationAvgLifeTime
    global pos

    nSurvivors = 0
    avgLifeTime = 0
    mutation_probability = 0.01
    energy = 0

    # For each individual you can extract the following information left over
    # from evaluation to let you figure out how well individual did in the
    # simulation of the world: whether the creature is dead or not, how much
    # energy did the creature have a the end of simulation (0 if dead), tick number
    # of creature's death (if dead).  You should use this information to build
    # a fitness function, score for how the individual did
    for individual in old_population:

        # You can read the creature's energy at the end of the simulation.  It will be 0 if creature is dead
        energy += individual.getEnergy()

        # This method tells you if the creature died during the simulation
        dead = individual.isDead()

        # If the creature is dead, you can get its time of death (in turns)
        if dead:
            timeOfDeath = individual.timeOfDeath()
            avgLifeTime += timeOfDeath
        else:
            nSurvivors += 1
            avgLifeTime += numTurns

        individual.fitness = fitness(individual)

    # Here are some statistics, which you may or may not find useful
    avgLifeTime = float(avgLifeTime)/float(len(population))
    generationAvgLifeTime[pos] = avgLifeTime
    avgFitness = population_fitness(old_population)/float(len(population))
    generationFitness[pos] = avgFitness
    pos += 1

    print("Simulation stats:")
    print("  Survivors    : %d out of %d" % (nSurvivors, len(population)))
    print("  Avg life time: %.1f turns" % avgLifeTime)
    print("  Avg population fitness: %d" % avgFitness)

    # The information gathered above should allow you to build a fitness function that evaluates fitness of
    # every creature.  You should show the average fitness, but also use the fitness for selecting parents and
    # creating new creatures.

    # Based on the fitness you should select individuals for reproduction and create a
    # new population.  At the moment this is not done, and the same population with the same number
    # of individuals
    # new_population = old_population

    new_population = []

    elite = pick_elite(old_population)
    for best in elite:
        new_population.append(best)

    while len(new_population) < len(population) - 1:
        parents = select_parents(all_creatures=old_population)

        child_1, child_2 = crossover(parents=parents)
        new_population.append(mutate(individual=child_1, mutate_prob=mutation_probability))
        new_population.append(mutate(individual=child_2, mutate_prob=mutation_probability))

    return new_population


def plot():

    plt.figure(1)

    plt.subplot(211)
    plt.plot(generationFitness)
    plt.title("World: %d Grid: %d Generations: %d\n" % (worldType, gridSize, numGenerations))
    plt.ylabel("Average Fitness")

    plt.subplot(212)
    plt.plot(generationAvgLifeTime)
    plt.ylabel("Average Life Time")
    plt.xlabel("Generation number")

    plt.show()

plt.close('all')
fh = plt.figure()

# Create the world.  Representation type chooses the type of percept representation
# (there are three types to chose from);
# gridSize specifies the size of the world, repeatable parameter allows you to run the simulation in exactly same way.
w = World(worldType=worldType, gridSize=gridSize, repeatable=repeatableMode)

# Get the number of creatures in the world
numCreatures = w.maxNumCreatures()

# Get the number of creature percepts
numCreaturePercepts = w.numCreaturePercepts()

# Get the number of creature actions
numCreatureActions = w.numCreatureActions()

# Create a list of initial creatures - instantiations of the MyCreature class that you implemented
population = list()
for i in range(numCreatures):
    c = MyCreature(numCreaturePercepts, numCreatureActions)
    population.append(c)

# Pass the first population to the world simulator
w.setNextGeneration(population)

# Runs the simulation to evalute the first population
w.evaluate(numTurns)

# Show visualisation of initial creature behaviour
w.show_simulation(titleStr='Initial population', speed='fast')

for i in range(numGenerations):
    print("\nGeneration %d:" % (i+1))

    # Create a new population from the old one
    population = newPopulation(population)

    # Pass the new population to the world simulator
    w.setNextGeneration(population)

    # Run the simulation again to evalute the next population
    w.evaluate(numTurns)

    # Show visualisation of final generation
    if i == numGenerations-1:
        w.show_simulation(titleStr='Final population', speed='normal')

plot()

