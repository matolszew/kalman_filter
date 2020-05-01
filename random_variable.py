import numpy as np
import matplotlib.pyplot as plt
import argparse

from kalman_filter import KalmanFilter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate expected value of the random variable")
    parser.add_argument('-m', '--mean', type=int, default=1, help='mean value of the random variable')
    parser.add_argument('-v', '--var', type=int, default=1, help='variance of the random variable')
    parser.add_argument('-N', '--samples', type=int, default=50, help='number of samples')
    parser.add_argument('-o', '--out_file', type=str, default='out.png', help='name of the output file')
    args = parser.parse_args()

    mean = args.mean
    var = args.var
    n = args.samples

    kf = KalmanFilter(
        state=np.array([[0]]),
        covariance=np.array([[var]]),
        state_transition=np.array([[1]]),
        control_input=np.array([[0]]),
        observation=np.array([[1]]),
        process_noise=np.array([[0]]),
        observation_noise=np.array([[var]])
        )

    state = np.zeros((N))
    y = np.zeros((N))
    for i in range(N):
        y[i] = np.array([mean]) + var*np.random.randn(1)
        state[i], _ = kf.estimate(y[i], np.array([[0]]))

    fig = plt.figure()
    plt.title('Expected value of the random variable')
    plt.plot(mean*np.ones((N)), 'g--', label='expected value')
    plt.plot(state, 'r', label='estimated expected value')
    plt.plot(y, 'x', label='measurments')
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(args.out_file)
