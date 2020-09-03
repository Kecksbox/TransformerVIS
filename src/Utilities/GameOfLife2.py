# Python code to implement Conway's Game Of Life
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from random import randrange, randint, seed
import tensorflow as tf

seed(0)
np.random.seed(0)

class Game:

    def __init__(self, initPlayerPos):
        self.playerPos = initPlayerPos
        self.satiety = 8
        self.poisonCounter = -1
        self.appels = [
            (1, 4),
            (2, 3),
            (1, 2),
            (0, 1),
            (2, 0),
        ]
        self.poisendAppels = [
            (4, 1)
        ]
        self.grave = [0, 4]
        self.resort = (0, 0)

    def observe(self):
        grid = np.zeros((5, 5))

        grid[self.grave[0], self.grave[1]] = 4
        grid[self.resort[0], self.resort[1]] = 5

        for apple in self.appels:
            grid[apple[0], apple[1]] = 2
        for apple in self.poisendAppels:
            grid[apple[0], apple[1]] = 3

        grid[self.playerPos[0], self.playerPos[1]] = 20

        return grid, self.playerPos

    def act(self, action):

        movement = [0, 0]
        if action == 0:
            movement = [-1, 0]
        elif action == 1:
            movement = [0, 1]
        elif action == 2:
            movement = [1, 0]
        elif action == 3:
            movement = [0, -1]


        if self.resort[0] == self.playerPos[0] and self.resort[1] == self.playerPos[1]:
            return

        if self.grave[0] == self.playerPos[0] and self.grave[1] == self.playerPos[1]:
            return

        self.playerPos[0] = min(4, max(self.playerPos[0] + movement[0], 0))
        self.playerPos[1] = min(4, max(self.playerPos[1] + movement[1], 0))

        self.satiety -= 1
        if self.poisonCounter > 0:
            self.poisonCounter -= 1

        # check for apples
        for apple in self.appels:
            if apple[0] == self.playerPos[0] and apple[1] == self.playerPos[1]:
                # remove apple
                self.appels.remove(apple)
                self.satiety = 8

        for apple in self.poisendAppels:
            if apple[0] == self.playerPos[0] and apple[1] == self.playerPos[1] and self.poisonCounter < 0:
                self.poisendAppels.remove(apple)
                self.poisonCounter = 2

        # check for death
        if self.poisonCounter == 0 or self.satiety == 0:
            self.playerPos = self.grave

def randomParameterGrid(N):
    return np.random.randint(4, size=(N, N))

def simulate(length):
    run = [None] * length
    game = Game([randrange(5), randrange(5)])
    paramterGrid = randomParameterGrid(5)
    initalState, _ = game.observe()
    for i in range(length):
        run[i], playerPos = game.observe()
        game.act(paramterGrid[playerPos[0], playerPos[1]])

    parameters = tf.stack([paramterGrid, initalState], axis=2)
    return (
        tf.expand_dims(parameters, axis=-1),
        tf.expand_dims(tf.expand_dims(run, axis=-1), axis=-1),
    )

def show_internal(frameNum, img, run):
    new_grid = tf.squeeze(run[frameNum])
    img.set_data(new_grid)
    return img,


def show(run, interval=200):
    updateInterval = int(interval)

    grid = tf.squeeze(run[0])

    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, show_internal, fargs=(img, run),
                                  frames=run.__len__(),
                                  interval=updateInterval,
                                  save_count=50)

    plt.show()


def createTestSet_internal():
    for _ in range(4000):
        yield simulate(20)

def createTestSet():
    return tf.data.Dataset.from_generator(
        createTestSet_internal,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([5, 5, 2, 1]), tf.TensorShape([None, 5, 5, 1, 1]))
    )


if __name__ == '__main__':
    testRun = simulate(10)
    testSet = createTestSet()
    for e in testSet:
        show(e[1])



