from enum import Enum
from copy import deepcopy
import os
import random

# Global variables
Y_POSITION_MAX = 20
X_POSITION_MAX = 20


class Direction(Enum):
    VERTICAL = 1
    HORIZONTAL = 2


class Graph:
    def __init__(self, n):
        self.n = n
        self.graph = [[0 for _ in range(n)] for _ in range(n)]

    def add_edge(self, u, v):
        self.graph[u][v] = 1
        self.graph[v][u] = 1

    def is_connected(self):
        visited = [False for _ in range(self.n)]
        self.__dfs(0, visited)
        for i in range(self.n):
            if not visited[i]:
                return False
        return True

    def number_of_disconnected_nodes(self):
        visited = [False for _ in range(self.n)]
        self.__dfs(0, visited)
        count = 0
        for i in range(self.n):
            if not visited[i]:
                count += 1
        return count

    def __dfs(self, u, visited):
        visited[u] = True
        for v in range(self.n):
            if self.graph[u][v] == 1 and not visited[v]:
                self.__dfs(v, visited)


class Gene:
    """
    This class represents a gene in the chromosome. Each gene is a word with its position and direction.
    """
    def __init__(self, word, x_position, y_position, direction):
        """
        :param word: string representing the word
        :param x_position: integer representing the x position of the word
        :param y_position: integer representing the y position of the word
        :param direction: Direction enum representing the direction of the word. Could be vertical or horizontal
        """
        self.word = word
        self.x_position = x_position
        self.y_position = y_position
        self.direction = direction

    def __str__(self):
        return f"Word: {self.word}, X: {self.x_position}, Y: {self.y_position}, Direction: {self.direction}"


class Chromosome:
    """
    This class represents a chromosome in the population. Each chromosome is a crossword puzzle.
    """
    def __init__(self, genes):
        """
        :param genes: list of genes representing the words in the crossword puzzle
        """
        self.genes = genes
        self.fitness = 1000
        self.update_fitness()

    def print_cage(self):
        """
        This function prints the crossword cage
        """
        cage = [[" " for _ in range(X_POSITION_MAX)] for _ in range(Y_POSITION_MAX)]
        for gene in self.genes:
            for i in range(len(gene.word)):
                if gene.direction == Direction.VERTICAL:
                    cage[gene.y_position + i][gene.x_position] = gene.word[i]
                else:
                    cage[gene.y_position][gene.x_position + i] = gene.word[i]
        for row in cage:
            print(row)
        print("\n")


    def update_fitness(self):
        """
        This function updates the fitness of the chromosome
        """
        # Graph is needed to check if the words are intersecting
        g = Graph(len(self.genes))

        fitness = 0

        for i in range(len(self.genes)):
            # check if the words are out of bounds
            if self.genes[i].direction == Direction.VERTICAL:
                if self.genes[i].y_position < 0 or self.genes[i].y_position + len(self.genes[i].word) >= Y_POSITION_MAX:
                    fitness += 10
                if self.genes[i].x_position < 0 or self.genes[i].x_position >= X_POSITION_MAX:
                    fitness += 10
            else:
                if self.genes[i].x_position < 0 or self.genes[i].x_position + len(self.genes[i].word) >= X_POSITION_MAX:
                    fitness += 10
                if self.genes[i].y_position < 0 or self.genes[i].y_position >= Y_POSITION_MAX:
                    fitness += 10

            for j in range(i + 1, len(self.genes)):
                # if both words are vertical
                if self.genes[i].direction == Direction.VERTICAL and self.genes[j].direction == Direction.VERTICAL:
                    # firstly check if they are in the +-1 x column
                    if abs(self.genes[i].x_position - self.genes[j].x_position) <= 1:
                        # if they have collision, then add error
                        if self.genes[i].y_position <= self.genes[j].y_position < self.genes[i].y_position + len(
                                self.genes[i].word):
                            fitness += 1
                        # if they have collision, then add error
                        elif self.genes[j].y_position <= self.genes[i].y_position < self.genes[j].y_position + len(
                                self.genes[j].word):
                            fitness += 1

                # if both words are horizontal
                elif self.genes[i].direction == Direction.HORIZONTAL and self.genes[j].direction == Direction.HORIZONTAL:
                    # firstly check if they are in the +-1 y column
                    if abs(self.genes[i].y_position - self.genes[j].y_position) <= 1:
                        # if they have collision, then add error
                        if self.genes[i].x_position <= self.genes[j].x_position < self.genes[i].x_position + len(
                                self.genes[i].word):
                            fitness += 1
                        # if they have collision, then add error
                        elif self.genes[j].x_position <= self.genes[i].x_position < self.genes[j].x_position + len(
                                self.genes[j].word):
                            fitness += 1




                # if the first word is vertical and the second is horizontal
                elif self.genes[i].direction == Direction.VERTICAL and self.genes[j].direction == Direction.HORIZONTAL:
                    if self.genes[j].x_position <= self.genes[i].x_position < self.genes[j].x_position + len(
                            self.genes[j].word):
                        if self.genes[i].y_position <= self.genes[j].y_position < self.genes[i].y_position + len(
                                self.genes[i].word):
                            # if they intersect then add them to the graph
                            g.add_edge(i, j)
                            g.add_edge(j, i)
                            # check if the words are intersecting in the same letter
                            if self.genes[i].word[self.genes[j].y_position - self.genes[i].y_position] != \
                                    self.genes[j].word[self.genes[i].x_position - self.genes[j].x_position]:
                                # if the letters are not the same then add error
                                fitness += 1

                    ii, jj = i, j
                    # top collision
                    if self.genes[jj].y_position == self.genes[ii].y_position - 1:
                        if self.genes[jj].x_position <= self.genes[ii].x_position < self.genes[jj].x_position + len(
                                self.genes[jj].word):
                            fitness += 1

                    # bottom collision
                    if self.genes[jj].y_position == self.genes[ii].y_position + len(self.genes[ii].word):
                        if self.genes[jj].x_position <= self.genes[ii].x_position < self.genes[jj].x_position + len(
                                self.genes[jj].word):
                            fitness += 1

                    # left collision
                    if self.genes[jj].x_position + len(self.genes[jj].word) - 1 == self.genes[ii].x_position - 1:
                        if self.genes[ii].y_position <= self.genes[jj].y_position <= self.genes[ii].y_position + len(
                                self.genes[ii].word) - 1:
                            fitness += 1

                    # right collision
                    if self.genes[jj].x_position == self.genes[ii].x_position + 1:
                        if self.genes[ii].y_position <= self.genes[jj].y_position <= self.genes[ii].y_position + len(
                                self.genes[ii].word) - 1:
                            fitness += 1

                # if the first word is horizontal and the second is vertical
                elif self.genes[i].direction == Direction.HORIZONTAL and self.genes[j].direction == Direction.VERTICAL:
                    if self.genes[i].x_position <= self.genes[j].x_position < self.genes[i].x_position + len(
                            self.genes[i].word):
                        if self.genes[j].y_position <= self.genes[i].y_position < self.genes[j].y_position + len(self.genes[j].word):
                            # if they intersect then add them to the graph
                            g.add_edge(i, j)
                            g.add_edge(j, i)
                            # check if the words are intersecting in the same letter
                            if self.genes[j].word[self.genes[i].y_position - self.genes[j].y_position] != \
                                    self.genes[i].word[self.genes[j].x_position - self.genes[i].x_position]:
                                fitness += 1

                    jj, ii = i, j
                    # top collision
                    if self.genes[jj].y_position == self.genes[ii].y_position - 1:
                        if self.genes[jj].x_position <= self.genes[ii].x_position < self.genes[jj].x_position + len(
                                self.genes[jj].word):
                            fitness += 1

                    # bottom collision
                    if self.genes[jj].y_position == self.genes[ii].y_position + len(self.genes[ii].word):
                        if self.genes[jj].x_position <= self.genes[ii].x_position < self.genes[jj].x_position + len(
                                self.genes[jj].word):
                            fitness += 1

                    # left collision
                    if self.genes[jj].x_position + len(self.genes[jj].word) - 1 == self.genes[ii].x_position - 1:
                        if self.genes[ii].y_position <= self.genes[jj].y_position <= self.genes[ii].y_position + len(
                                self.genes[ii].word) - 1:
                            fitness += 1

                    # right collision
                    if self.genes[jj].x_position == self.genes[ii].x_position + 1:
                        if self.genes[ii].y_position <= self.genes[jj].y_position <= self.genes[ii].y_position + len(
                                self.genes[ii].word) - 1:
                            fitness += 1

        # add error for each disconnected word
        fitness += g.number_of_disconnected_nodes()

        # update chromosome fitness
        self.fitness = fitness

    def __str__(self):
        string = ""
        for gene in self.genes:
            string += str(gene) + "\n"
        return string


class Crossword:
    """
    This class is used to create a crossword puzzle using a genetic algorithm.
    """
    def __init__(self):
        self.population = None
        self.words = None
        self.original_words = None

    def run_ga(self):
        """
        This function runs the genetic algorithm with optimization
        """
        # create initial population with 2 words in each chromosome
        self.create_population(2)
        i = 1
        # Run the genetic algorith for 2 words, then 3 words, etc...
        while i < len(self.words):
            print(f"Running GA with {i+1} words")
            # If the perfect crossword is found
            if self.ga():
                if i == len(self.words) - 1:
                    return
                # add one more word to the whole population
                for chromosome in self.population:
                    direction = random.choice(list(Direction))
                    if direction == Direction.VERTICAL:
                        x_position = random.randint(0, X_POSITION_MAX - 1)
                        y_position = random.randint(0, Y_POSITION_MAX - len(self.words[i+1]))
                    else:
                        x_position = random.randint(0, X_POSITION_MAX - len(self.words[i+1]))
                        y_position = random.randint(0, Y_POSITION_MAX - 1)
                    new_gene = Gene(self.words[i+1], x_position, y_position, direction)
                    chromosome.genes.append(new_gene)
                    chromosome.update_fitness()
                i += 1
            else:
                # start over
                self.population = None
                self.create_population(2)
                i = 1

    def ga(self, max_generation=20):
        """
        This function runs the genetic algorithm
        :param max_generation: number of maximum generations
        :return: True if the perfect crossword is found, False otherwise
        """
        best_fitness = float('inf')
        # number of generations when the best fitness is not changing
        stagnant_generations = 0

        for i in range(max_generation):
            # update the fitness of each chromosome
            for chromosome in self.population:
                chromosome.update_fitness()
            # sort the population by fitness
            self.population.sort(key=lambda x: x.fitness)
            # update the best fitness
            current_best_fitness = self.population[0].fitness
            print(f"Generation {i+1}: Best fitness {current_best_fitness}")

            # check if the best fitness is not changing
            if current_best_fitness == best_fitness:
                stagnant_generations += 1
            else:
                best_fitness = current_best_fitness
                stagnant_generations = 0

            # shake the population if the best fitness is not changing
            if stagnant_generations >= 5 and best_fitness > 0:
                self.mutation(self.population, 0.05)
            # if the best fitness is found, then wait for 5 generations to fill the population with more
            # perfect crosswords
            elif stagnant_generations >= 5 and best_fitness == 0:
                break
            # if the final crossword is found, then stop
            if current_best_fitness == 0 and len(self.population[0].genes) == len(self.words):
                self.population[0].print_cage()
                return True
            # perform selection with mutations and update the population
            self.__selection()

        # when the generation is over, sort the population by fitness
        for chromosome in self.population:
            chromosome.update_fitness()
        # sort the population by fitness
        self.population.sort(key=lambda x: x.fitness)
        # if the perfect crossword is not found, then return False
        if self.population[0].fitness != 0:
            return False
        # return True otherwise
        return True

    def input_words(self, file_path):
        """
        This function reads the words from the input file
        :param file_path: path to the input file
        :return:
        """
        try:
            with open(file_path, 'r') as file:
                self.words = [line.strip() for line in file if line.strip()]
        except Exception as e:
            print("An error occurred: {e}")

        # save the words original order
        self.original_words = deepcopy(self.words)

        # Some optimization
        # sort the words by length. Loger words are more likely to be intersected
        self.words.sort(key=len, reverse=True)

    def create_population(self, n, population_size=1000):
        """
        This function creates the initial population
        :param n: number of words in each chromosome
        :param population_size: number of chromosomes in the population
        """
        self.population = []
        for i in range(population_size):
            self.population.append(self.__create_chromosome(n))

    def __create_chromosome(self, n):
        """
        This function generates a random chromosome with n words
        :param n: number of genes in the chromosome
        """
        genes = []
        for word in self.words[:n]:
            direction = random.choice(list(Direction))
            if direction == Direction.VERTICAL:
                x_position = random.randint(0, X_POSITION_MAX - 1)
                y_position = random.randint(0, Y_POSITION_MAX - len(word))
            else:
                x_position = random.randint(0, X_POSITION_MAX - len(word))
                y_position = random.randint(0, Y_POSITION_MAX - 1)
            new_gene = Gene(word, x_position, y_position, direction)
            genes.append(new_gene)
        return Chromosome(genes)

    def __selection(self, best_individuals_percentage=0.2):
        """
        This function performs selection, crossover and mutation. It also updates the population
        :param best_individuals_percentage: percentage of the best individuals in the population
        to be selected for the next generation
        """
        # select the best individuals
        best_individuals = self.population[:int(len(self.population) * best_individuals_percentage)]

        # select the rest of the individuals randomly
        rest_individuals_len = len(self.population) - int(len(self.population) * best_individuals_percentage)
        rest_individuals = random.sample(self.population[:], rest_individuals_len)

        # perform crossover of all the individuals by pairing them randomly
        new_individuals = []
        for i in range(len(rest_individuals)):
            new_individuals.append(self.__crossover(self.tournament_selection(2), self.tournament_selection(2)))

        # perform mutation on the new individuals
        self.mutation(new_individuals, 0.01)

        # update the population
        new_population = best_individuals + new_individuals
        self.population = deepcopy(new_population)

    def tournament_selection(self, tournament_size):
        """
        This function performs tournament selection
        :param tournament_size: number of random individuals in the tournament
        :return:
        """
        # Initialize an empty list for the tournament
        tournament = []

        # Select random individuals for the tournament
        for _ in range(tournament_size):
            tournament.append(random.choice(self.population))

        # Sort the tournament individuals by fitness
        tournament.sort(key=lambda x: x.fitness)

        # Return the best individual
        return tournament[0]


    def __crossover(self, parent1, parent2):
        """
        This function performs crossover of two parents
        :param parent1: first chromosome to be crossed
        :param parent2: second chromosome to be crossed
        :return: new chromosome
        """
        # select two random points in the chromosome
        i1 = random.randint(0, len(parent1.genes) - 1)
        i2 = random.randint(0, len(parent1.genes) - 1)
        if i1 > i2:
            i1, i2 = i2, i1
        new_genes = deepcopy(parent1.genes)
        # perform crossover of the middle part of the chromosome
        for i in range(i1, i2 + 1):
            new_genes[i] = deepcopy(parent2.genes[i])
        return Chromosome(new_genes)

    def output(self, file_path):
        """
        This function outputs the crossword puzzle to the file
        :param file_path: path to the output file
        """
        # write the best chromosome to the file
        with open(file_path, 'w') as file:
            for word in self.original_words:
                for gene in self.population[0].genes:
                    if word == gene.word:
                        file.write(f"{gene.y_position} {gene.x_position} {1 if gene.direction == Direction.VERTICAL else 0}\n")
                        break

    def mutation(self, selection, mutation_rate):
        """
        This function performs mutation of the selected individuals
        :param selection: list of selected individuals
        :param mutation_rate: probability of mutation
        """
        # for each gene in each chromosome mutate a gene with probability mutation_rate
        for chromosome in selection:
            for gene in chromosome.genes:
                if random.random() < mutation_rate:
                    if gene.direction == Direction.VERTICAL:
                        gene.x_position = random.randint(0, X_POSITION_MAX - len(gene.word))
                    else:
                        gene.x_position = random.randint(0, X_POSITION_MAX - 1)
                if random.random() < mutation_rate:
                    if gene.direction == Direction.HORIZONTAL:
                        gene.y_position = random.randint(0, Y_POSITION_MAX - len(gene.word))
                    else:
                        gene.y_position = random.randint(0, Y_POSITION_MAX - 1)
                if random.random() < mutation_rate:
                    gene.direction = random.choice(list(Direction))
                    # check if the word is out of bounds
                    if gene.direction == Direction.VERTICAL:
                        if gene.y_position + len(gene.word) > Y_POSITION_MAX:
                            gene.y_position = random.randint(0, Y_POSITION_MAX - len(gene.word))
                    else:
                        if gene.x_position + len(gene.word) > X_POSITION_MAX:
                            gene.x_position = random.randint(0, X_POSITION_MAX - len(gene.word))


def main():
    # create the directory for the outputs
    if not os.path.exists("outputs"):
        os.mkdir("outputs")

    # for each input file create the crossword puzzle
    for i in range(len(os.listdir("inputs"))):
        print(f"Running test {i+1}", end=" ")
        crossword = Crossword()
        # read the words from the input file
        crossword.input_words(f"inputs/input{i+1}.txt")
        print(f"with {len(crossword.words)} words")
        crossword.run_ga()
        # output the crossword puzzle to the file
        crossword.output(f"outputs/output{i+1}.txt")
        i += 1

main()