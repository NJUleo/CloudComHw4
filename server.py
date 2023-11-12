import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

"""
    Feedback structure code. You can use multiple source-codes for each of the controllers.
    Import all the source-codes in this code
"""
import feedback as fb


class AbstractServerPool:
    def __init__(self, n, server, client):
        self.n = n  # number of server instances to start with
        self.n_requests = 0  # number of arriving requests

        self.server = server  # server work function
        self.client = client  # queue-loading work function

    # set the number of server instances to u, and process work for this epoch
    # return the number of request completed in this epoch.
    def work(self, u):
        """simulate one epoch

        Args:
            u (int): number of server instances in this epoch

        Returns:
            int: number of requests completed
        """
        self.n = max(0, int(round(u)))  # server count: non-negative integer

        completed = 0
        for _ in range(self.n):
            completed += self.server()  # each server does some amount of work

            if completed >= self.n_requests:
                completed = (
                    self.n_requests
                )  # "trim" completed to total requests arrived
                break  # stop if queue is empty

        self.n_requests -= completed  # reduce total requests by work completed

        return completed

    def monitoring(self):
        return "To be implemented"


# ============================================================
# Server Pool
class ServerPool(AbstractServerPool):
    def work(self, u):
        """simulate requests generation and handling in an epoch

        Args:
            u (int): number of server used in the epoch

        Returns:
            float: completion rate
        """
        load = self.client()  # generate new requests
        self.n_requests = (
            load  # new load replaces old load, because unhandled requests are dropped.
        )

        if load == 0:
            return 1  # no work: 100 percent completion rate

        completed = super().work(u)

        return completed / load  # completion rate


# Generate and Complete functions
def generate_work():
    """generate work for the closed loop system

    Returns:
        int: # of new requests
    """
    global global_time
    global_time += 1

    if global_time > 2500:
        return random.gauss(1600, 10)

    if global_time > 2200:
        return random.gauss(1000, 10)

    return random.gauss(1300, 10)


def complete_work():
    """
    return the number of requests processed by a server in an epoch.

    This method simulates the server aspects of the target system. This method
    simulates the processing of requests by generating a random number when invoked. This number
    represents the total number of requests processed by the server at every epoch.
    """
    a, b = 20, 3
    return 100 * random.betavariate(a, b)  # mean: a/(a+b); var: ~b/a^2


# ============================================================


def static_test(traffic):
    """
    Method to model the target system
    This method simulates the expected behavior of the target system used in this
    assignment. You will use this method to model the target system.
    """

    def generate_work():
        """
        This method simulates the client aspects of the target system. This method
        simulates work by generating a random number when invoked. This number represents the total
        number of requests that arrive at the server at every epoch.
        """
        return random.gauss(traffic, traffic / 200)

    # TODO: why are we using this generate_work method?
    samples = fb.static_test(
        ServerPool, (0, complete_work, generate_work), 20, 20, 10, 1
    )  # max u, steps, trials, timesteps

    return samples


def sine_static_test(traffic, repeats):
    """model the system with sine input

    Args:
        traffic (int): number of requests(for the gaussian)
    """

    def generate_work():
        return random.gauss(traffic, traffic / 200)

    samples = fb.static_test_sine(
        ServerPool, (0, complete_work, generate_work), 0, 12, repeats
    )

    return samples


def closed_loop(n):
    """
    Method to simulate the feedback-based system
    """
    initial_num_servers = n
    target_system = ServerPool(initial_num_servers, complete_work, generate_work)
    # controller = Create controller instance
    # Invoke closed_loop method in the controller


# ============================================================
# Helper Methods
def plotter():
    # Your plotter code should go here
    return


# If you are using any other helper methods, include them here


def save_object(obj, filename):
    """Save an object to a file."""
    with open(filename, "wb") as outp:  # Open the file in binary write mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Read an object from a file."""
    with open(filename, "rb") as inp:  # Open the file in binary read mode
        return pickle.load(inp)


def save_tuples_list(lst, filename):
    """Save a list of tuples to a text file."""
    with open(filename, "w") as file:
        for item in lst:
            file.write(f"{item[0]}, {item[1]}\n")  # Write each tuple as 'x, y'


def load_tuples_list(filename):
    """Read a list of tuples from a text file."""
    with open(filename, "r") as file:
        return [tuple(map(float, line.strip().split(", "))) for line in file]


def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE).

    Parameters:
    y_true (list): The true values.
    y_pred (list): The predicted values.

    Returns:
    float: The RMSE value.
    """

    # Convert lists to numpy arrays to perform element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse_value = np.sqrt(mse)

    return rmse_value


def r_squared(y_true, y_pred):
    """
    Calculate the coefficient of determination, R^2.

    Parameters:
    y_true (list): The true values.
    y_pred (list): The predicted values.

    Returns:
    float: The R^2 value.
    """

    # Convert lists to numpy arrays to perform element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the variance of the residuals
    residual_variance = np.var(y_true - y_pred)

    # Calculate the variance of the original values
    true_variance = np.var(y_true)

    # Calculate R^2
    r2 = 1 - (residual_variance / true_variance)

    return r2


def plot_samples(samples):
    u, y = zip(*samples)

    # Plotting the points
    plt.scatter(u, y)
    plt.title("Sample Data Points")
    plt.xlabel("u values")
    plt.ylabel("y values")
    plt.grid(True)
    plt.show()


def model(samples):
    # Sample data (u, y)
    u = [u for u, _ in samples]
    y = [y for _, y in samples]
    yn = [0] + y[:-1]

    # Extract y(n) and u(n) from data
    y_n_plus1 = np.array(y)
    u_n = np.array(u)
    y_n = np.array(yn)

    # Create the design matrix
    X = np.column_stack((y_n, u_n))

    # Perform linear regression
    a, b = np.linalg.lstsq(X, y_n_plus1, rcond=None)[0]

    # Now a and b are the coefficients of your linear regression model
    print(f"a: {a}, b: {b}")

    return a, b


# ============================================================


if __name__ == "__main__":
    """
    TA will only type "python server.py k N T C"

    k - Number of time steps to simulate
    N - Number of initial server instances
    T - Type of test (s, c) - s = static test, c = simulate feedback-based system
    C - Controller to run (p, pi, pid)
    p = proportional controller; pi = proportional-integral controller; pid = proportional-integral-derivative controller
    Note: You must handle any errors in the user input
    """

    # TODO: what's this DT? nothing used there.
    fb.DT = 1  # Sampling time is set to 1 - Refer to feedback.py

    global_time = 0  # To communicate with generate and consume functions

    # consturct sampel and model
    do_sample = False
    samples = None
    sample_file_name = "sine_samples.txt"

    if do_sample:
        samples = sine_static_test(1000, 1000)
        plot_samples(samples)
        save_tuples_list(samples, sample_file_name)
    else:
        # load sample from file
        samples = load_tuples_list(sample_file_name)

    a, b = model(samples)

    # test model
    do_test = False
    test_samples = None
    test_sample_file_name = "test_sine_samples.txt"

    if do_test:
        test_samples = sine_static_test(1000, 200)
        plot_samples(test_samples)
        save_tuples_list(test_samples, test_sample_file_name)
    else:
        # load sample from file
        test_samples = load_tuples_list(test_sample_file_name)

    y_true = [y for _, y in test_samples]
    y_pred = []
    y_prev = 0
    for u, _ in test_samples:
        y_cur = a * y_prev + b * u
        y_prev = y_cur
        y_pred.append(y_cur)

    test_rmse = rmse(y_true, y_pred)
    test_r2 = r_squared(y_true, y_pred)
    print(f"RMSE: {test_rmse}, r^2: {test_r2}")

    print("haha")


#    closed_loop()
