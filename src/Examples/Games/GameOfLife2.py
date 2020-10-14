# Python code to implement Conway's Game Of Life
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from random import randrange, randint, seed
import tensorflow as tf

seed(0)
np.random.seed(0)

size = 500

class Game:

    def __init__(self, initPlayerPos):
        self.playerPos = initPlayerPos
        self.satiety = 6
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
        grid = np.zeros((size, size))

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
                self.satiety = 6

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
    game = Game([randrange(size), randrange(size)])
    paramterGrid = randomParameterGrid(size)
    initalState, _ = game.observe()
    for i in range(length):
        run[i], playerPos = game.observe()
        game.act(paramterGrid[playerPos[0], playerPos[1]])

    parameters = tf.stack([paramterGrid, initalState], axis=2)
    return (
        tf.expand_dims(parameters, axis=-1),
        tf.expand_dims(tf.expand_dims(run, axis=-1), axis=-1),
    )

def show_internal(frameNum, imges, runs, latents, latentplot):
    for i in range(3):
        # time_text.set_text('%.1d' % frameNum)
        new_grid = tf.squeeze(runs[i][frameNum])
        imges[i].set_data(new_grid)
    latentplot.cla()
    for latent in latents:
        latent = tf.squeeze(latent, axis=0)
        latentplot.plot(latent[:frameNum,1], latent[:frameNum,0])

counter = 0

def show(runs, latents, interval=200, save=False):
    updateInterval = int(interval)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    colors = ['blue', 'orange', 'green']
    images = []
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0])
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.patch.set_edgecolor(colors[i])
        ax.patch.set_linewidth('4')
        grid = tf.squeeze(runs[i][0])
        #time_text = axes[i].text(0.05, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=axes[i].transAxes)
        #plt.setp(time_text, color='w')
        img = ax.imshow(grid, interpolation='nearest')
        images.append(img)

    ax = fig.add_subplot(gs[:, 1:])
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    latentplot = ax

    for latent in latents:
        latent = tf.squeeze(latent, axis=0)
        latentplot.plot(latent[:0, 1], latent[:0, 0])

    ani = animation.FuncAnimation(fig, show_internal, fargs=(images, runs, latents, latentplot),
                                  frames=runs[0].__len__(),
                                  interval=updateInterval,
                                  save_count=100)

    plt.tight_layout()
    plt.show()

    if save:
        global counter
        counter += 1
        plt.rcParams['animation.ffmpeg_path'] = 'E:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe'
        mywriter = animation.FFMpegWriter()
        ani.save('Animated_Sequence.mp4'.format(counter), writer=mywriter)


def createTestSet_internal():
    for _ in range(3):
        yield simulate(100)

def createTestSet():
    return tf.data.Dataset.from_generator(
        createTestSet_internal,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([size, size, 2, 1]), tf.TensorShape([None, size, size, 1, 1]))
    )


if __name__ == '__main__':
    testRun = simulate(10)
    testSet = createTestSet()
    for e in testSet:
        show(e[1])



