import numpy as np

class KalmanFilter:
    """
    Args:
        state (np.array): initial state
        covariance (np.array): initial covariance matrix
        state_transition (np.array): matrix A
        control_input (np.array): matrix B
        observation (np.array): matrix C
        process_noise (np.array): covariance matrix of process noise
        observation_noise (np.array): covariance matrix of observation noise
    """
    def __init__(self, state, covariance, state_transition, control_input, observation, process_noise, observation_noise):
        self.state = state
        self.covariance = covariance
        self.state_transition = state_transition
        self.control_input = control_input
        self.observation = observation
        self.process_noise = process_noise
        self.observation_noise = observation_noise

    def estimate(self, y, u):
        """Calculate new state estimation

        Args:
            y (np.array): New measurment
            u (np.array): New input signal
        Returns:
            (tuple): tuple containing:

                state (np.array): State estimation
                covariance (np.array): Covariance matrix estimation
            """
        state_prediction = np.matmul(self.state_transition, self.state) + np.matmul(self.control_input, u)
        covariance_prediction = np.matmul(np.matmul(self.state_transition, self.covariance), self.state_transition.T) + self.process_noise

        innovation = y - np.matmul(self.observation, state_prediction)
        innovation_cov = np.matmul(np.matmul(self.observation, covariance_prediction), self.observation.T) + self.observation_noise
        gain = np.matmul(np.matmul(covariance_prediction, self.observation.T), np.linalg.inv(innovation_cov))
        self.state = state_prediction + np.matmul(gain, innovation)
        self.covariance = covariance_prediction - np.matmul(np.matmul(gain, innovation_cov), gain.T)

        return self.state, self.covariance
