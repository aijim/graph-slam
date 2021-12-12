import numpy as np
from helpers import make_data
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns

# your implementation of slam should work with the following inputs
# feel free to change these input values and see how it responds!

# world parameters
num_landmarks = 5        # number of landmarks
N = 20       # time steps
world_size = 100.0    # size of world (square)

# robot parameters
measurement_range = 50.0     # range at which we can sense landmarks
motion_noise = 2.0      # noise in robot motion
measurement_noise = 2.0      # noise in the measurements
distance = 20.0     # distance by which robot (intends to) move each iteratation


# make_data instantiates a robot, AND generates random landmarks for a given world size and number of landmarks
data = make_data(N, num_landmarks, world_size, measurement_range, motion_noise, measurement_noise, distance)


def initialize_constraints(N, num_landmarks, world_size):
    ''' This function takes in a number of time steps N, number of landmarks, and a world_size,
        and returns initialized constraint matrices, omega and xi.'''

    omega = np.zeros((2 * N + 2 * num_landmarks, 2 * N + 2 * num_landmarks))
    omega[0, 0] = 1
    omega[1, 1] = 1

    xi = np.zeros(2 * N + 2 * num_landmarks)
    xi[0] = world_size / 2
    xi[1] = world_size / 2

    return omega, xi

# define a small N and world_size (small for ease of visualization)
N_test = 5
num_landmarks_test = 2
small_world = 10

# initialize the constraints
initial_omega, initial_xi = initialize_constraints(N_test, num_landmarks_test, small_world)


## slam takes in 6 arguments and returns mu,
## mu is the entire path traversed by a robot (all x,y poses) *and* all landmarks locations
def slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise):
    omega, xi = initialize_constraints(N, num_landmarks, world_size)

    ## get all the motion and measurement data as you iterate
    for i in range(N - 1):
        measurements = data[i][0]
        motion = data[i][1]

        ## this should be a series of additions that take into account the measurement noise
        for measurement in measurements:
            index = measurement[0]
            dx = measurement[1]
            dy = measurement[2]

            omega[2 * i, 2 * i] += 1.0 / measurement_noise
            omega[2 * i, 2 * N + 2 * index] += -1.0 / measurement_noise
            omega[2 * N + 2 * index, 2 * i] += -1.0 / measurement_noise
            omega[2 * N + 2 * index, 2 * N + 2 * index] += 1.0 / measurement_noise
            omega[2 * i + 1, 2 * i + 1] += 1.0 / measurement_noise
            omega[2 * i + 1, 2 * N + 2 * index + 1] += -1.0 / measurement_noise
            omega[2 * N + 2 * index + 1, 2 * i + 1] += -1.0 / measurement_noise
            omega[2 * N + 2 * index + 1, 2 * N + 2 * index + 1] += 1.0 / measurement_noise

            xi[2 * i] += -dx / measurement_noise
            xi[2 * N + 2 * index] += dx / measurement_noise
            xi[2 * i + 1] += -dy / measurement_noise
            xi[2 * N + 2 * index + 1] += dy / measurement_noise

        omega[2 * i, 2 * i] += 1.0 / motion_noise
        omega[2 * i, 2 * (i + 1)] += -1.0 / motion_noise
        omega[2 * (i + 1), 2 * i] += -1.0 / motion_noise
        omega[2 * (i + 1), 2 * (i + 1)] += 1.0 / motion_noise
        omega[2 * i + 1, 2 * i + 1] += 1.0 / motion_noise
        omega[2 * i + 1, 2 * (i + 1) + 1] += -1.0 / motion_noise
        omega[2 * (i + 1) + 1, 2 * i + 1] += -1.0 / motion_noise
        omega[2 * (i + 1) + 1, 2 * (i + 1) + 1] += 1.0 / motion_noise

        xi[2 * i] += -motion[0] / motion_noise
        xi[2 * (i + 1)] += motion[0] / motion_noise
        xi[2 * i + 1] += -motion[1] / motion_noise
        xi[2 * (i + 1) + 1] += motion[1] / motion_noise

    ## Compute the best estimate of poses and landmark positions
    ## using the formula, omega_inverse * Xi
    mu = np.matmul(np.linalg.inv(omega), xi)

    return mu  # return `mu`

# a helper function that creates a list of poses and of landmarks for ease of printing
# this only works for the suggested constraint architecture of interlaced x,y poses
def get_poses_landmarks(mu, N):
    # create a list of poses
    poses = []
    for i in range(N):
        poses.append((mu[2*i].item(), mu[2*i+1].item()))

    # create a list of landmarks
    landmarks = []
    for i in range(num_landmarks):
        landmarks.append((mu[2*(N+i)].item(), mu[2*(N+i)+1].item()))

    # return completed lists
    return poses, landmarks

def print_all(poses, landmarks):
    print('\n')
    print('Estimated Poses:')
    for i in range(len(poses)):
        print('['+', '.join('%.3f'%p for p in poses[i])+']')
    print('\n')
    print('Estimated Landmarks:')
    for i in range(len(landmarks)):
        print('['+', '.join('%.3f'%l for l in landmarks[i])+']')

# call your implementation of slam, passing in the necessary parameters
mu = slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise)

# print out the resulting landmarks and poses
if(mu is not None):
    # get the lists of poses and landmarks
    # and print them out
    poses, landmarks = get_poses_landmarks(mu, N)
    print_all(poses, landmarks)