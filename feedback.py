import math
import random

SamplingInterval = None  # Sampling interval - defaults to None, must be set explicitly

# ============================================================
# Controllers

"""
    Include your controller(s) code here
"""


class p_controller:
    def __init__(self, kp) -> None:
        self.kp = kp

    def compute(self, e):
        return e * self.kp


class pi_controller:
    def __init__(self, kp, ki):
        self.kp = kp
        self.ki = ki
        self.ui_prev = 0

    def compute(self, e):
        ui = self.ui_prev + self.ki * e
        self.ui_prev = ui
        u = self.kp * e + ui
        return u


class pid_controller:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.e_prev = 0
        self.ui_prev = 0

    def compute(self, e):
        ui = self.ui_prev + self.ki * e
        self.ui_prev = ui
        ud = self.kd * (e - self.e_prev)
        self.e_prev = e
        u = self.kp * e + ui + ud
        return u


# ============================================================
# # Input signals

"""
    If you test your target system with various input signals, include
    the code here. You will only invoke the input signal that you 
    finalized, when you submit your code. 
"""


def sine_wave_values(lower_bound, upper_bound, num):
    """
    This function returns a list of integer values from a sine wave within a specified range.

    Parameters:
    lower_bound (float): The lower bound of the sine wave.
    upper_bound (float): The upper bound of the sine wave.
    num (int): The number of values to calculate.

    Returns:
    list: A list of integer values from the sine wave.
    """
    # Calculate the amplitude and mid-point of the sine wave
    amplitude = (upper_bound - lower_bound) / 2
    mid_point = (upper_bound + lower_bound) / 2

    # Generate the sine wave values
    sine_values = [
        int(round(amplitude * math.sin(k) + mid_point)) for k in range(1, num + 1)
    ]

    return sine_values


# # ============================================================
# # Loop functions


def static_test(target_ctor, ctor_args, umax, steps, repeats, tmax):
    """static test for a target system with parameters

    Args:
        target_ctor (serverpool): class of the system to be modeled
        ctor_args (tuple): (# of server at start, server work function, work load function)
        umax (int): maximum of u
        steps (int): # of steps to increase u(# of server) to umax.
        repeats (int): # of repeats for each step value of u.
        tmax (int): # of epoches for each repeat to make sure the completion rate is stable.

    Returns:
        [(u, y)]: samples
    """
    # Complete test for static process characteristic
    print("Inside FB ST")
    result = []
    # TODO: do we need to use this ramp signal to conduct the test?
    for i in range(0, steps):
        # number of server in each step.
        u = float(i) * umax / float(steps)

        for _ in range(repeats):
            # for each repeats, create a new instance of the server pool
            # that is, for each step, conduct several experiments.
            target = target_ctor(*ctor_args)

            for _ in range(tmax):
                # work for several epochs to make sure the completion rate is stable?
                y = target.work(u)

            print(u, y)
            result.append((u, y))

    return result


def static_test_sine(target_ctor, ctor_args, upper_bound, lower_bound, repeats):
    """static test for a target system with parameters

    Args:
        target_ctor (serverpool): class of the system to be modeled
        ctor_args (tuple): (# of server at start, server work function, work load function)
        umax (int): maximum of u
        steps (int): # of steps to increase u(# of server) to umax.
        repeats (int): # of repeats for each step value of u.
        tmax (int): # of epoches for each repeat to make sure the completion rate is stable.

    Returns:
        [(u, y)]: samples
    """
    # Complete test for static process characteristic
    print("static sine wave test")
    result = []
    inputs = sine_wave_values(upper_bound, lower_bound, repeats)
    for u in inputs:
        target = target_ctor(*ctor_args)
        y = target.work(u)
        result.append((u, y))

    return result
