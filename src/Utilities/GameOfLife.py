# Python code to implement Conway's Game Of Life
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

# setting up the values for the grid
ON = 255
OFF = 0
vals = [ON, OFF]

np.random.seed(0)

def randomGrid(N):
    """returns a grid of NxN random values"""
    return np.random.choice(vals, N * N, p=[0.2, 0.8]).reshape(N, N)


def addGlider(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[0, 0, 255],
                       [255, 0, 255],
                       [0, 255, 255]])
    grid[i:i + 3, j:j + 3] = glider


def addGosperGliderGun(i, j, grid):
    """adds a Gosper Glider Gun with top left
       cell at (i, j)"""
    gun = np.zeros(11 * 38).reshape(11, 38)

    gun[5][1] = gun[5][2] = 255
    gun[6][1] = gun[6][2] = 255

    gun[3][13] = gun[3][14] = 255
    gun[4][12] = gun[4][16] = 255
    gun[5][11] = gun[5][17] = 255
    gun[6][11] = gun[6][15] = gun[6][17] = gun[6][18] = 255
    gun[7][11] = gun[7][17] = 255
    gun[8][12] = gun[8][16] = 255
    gun[9][13] = gun[9][14] = 255

    gun[1][25] = 255
    gun[2][23] = gun[2][25] = 255
    gun[3][21] = gun[3][22] = 255
    gun[4][21] = gun[4][22] = 255
    gun[5][21] = gun[5][22] = 255
    gun[6][23] = gun[6][25] = 255
    gun[7][25] = 255

    gun[3][35] = gun[3][36] = 255
    gun[4][35] = gun[4][36] = 255

    grid[i:i + 11, j:j + 38] = gun


def update(grid, N):
    # copy grid since we require 8 neighbors
    # for calculation and we go line by line
    newGrid = grid.copy()
    for i in range(N):
        for j in range(N):

            # compute 8-neghbor sum
            # using toroidal boundary conditions - x and y wrap around
            # so that the simulaton takes place on a toroidal surface.
            total = int((grid[i, (j - 1) % N] + grid[i, (j + 1) % N] +
                         grid[(i - 1) % N, j] + grid[(i + 1) % N, j] +
                         grid[(i - 1) % N, (j - 1) % N] + grid[(i - 1) % N, (j + 1) % N] +
                         grid[(i + 1) % N, (j - 1) % N] + grid[(i + 1) % N, (j + 1) % N]) / 255)

            # apply Conway's rules
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    newGrid[i, j] = OFF
            else:
                if total == 3:
                    newGrid[i, j] = ON

                    # update data
    return newGrid


# main() function
def simulate(interval=50, length=200, grid_size=20, glider=False, gosper=False):
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored

    # set grid size
    if int(grid_size) > 8:
        N = int(grid_size)
    else:
        N = 10

    # declare grid
    grid = np.array([])

    # check if "glider" demo flag is specified
    grid = np.zeros(N * N).reshape(N, N)
    grid = randomGrid(N)
    if glider:
        addGlider(random.randint(1, N), random.randint(1, N), grid)
    if gosper:
        addGosperGliderGun(10, 10, grid)

    # set up animation
    run = [None] * length
    for i in range(length):
        run[i] = update(grid, N)
        grid = run[i]

    return run

def show_internal(frameNum, img, run):
    new_grid = tf.squeeze(run[frameNum])
    img.set_data(new_grid)
    return img,


def show(run, interval=50):
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
    for _ in range(4):
        yield tf.expand_dims(tf.expand_dims(simulate(grid_size=10, length=10), axis=-1), axis=-1)

def createTestSet(elementCount, grid_size=10, length=10):
    return tf.data.Dataset.from_generator(createTestSet_internal, output_types=tf.float32, output_shapes=(None, 10, 10, 1, 1))



# call main
if __name__ == '__main__':
    testSet = createTestSet(20)
    for e in testSet:
        show(e)
        break
