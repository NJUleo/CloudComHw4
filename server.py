import math
import random

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
        load = self.client()  # generate new requests
        self.n_requests = load  # new load replaces old load

        if load == 0:
            return 1  # no work: 100 percent completion rate

        # completed = AbstractServerPool.work( self, u )
        completed = super().work(u)

        return completed / load  # completion rate


# Generate and Complete functions
def generate_work():
    global global_time
    global_time += 1

    if global_time > 2500:
        return random.gauss(1600, 10)

    if global_time > 2200:
        return random.gauss(1000, 10)

    return random.gauss(1300, 10)


def complete_work():
    """
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

    fb.static_test(
        ServerPool, (0, complete_work, generate_work), 20, 20, 5, 1000
    )  # max u, steps, trials, timesteps


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

    fb.DT = 1  # Sampling time is set to 1 - Refer to feedback.py

    global_time = 0  # To communicate with generate and consume functions

    static_test(1000)
#    closed_loop()
