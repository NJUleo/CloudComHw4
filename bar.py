import matplotlib.pyplot as plt
from feedback import sine_wave_values


def plot_sine_wave_values(lower_bound, upper_bound, num):
    """
    This function plots integer values from a sine wave within a specified range.

    Parameters:
    lower_bound (float): The lower bound of the sine wave.
    upper_bound (float): The upper bound of the sine wave.
    num (int): The number of values to calculate.
    """
    # Get the sine wave values
    values = sine_wave_values(lower_bound, upper_bound, num)

    # Plot the values
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, num + 1), values, marker="o")
    plt.title(f"Sine Wave Values from {lower_bound} to {upper_bound} for k=1 to {num}")
    plt.xlabel("k")
    plt.ylabel("u values")
    plt.grid(True)
    plt.show()


# Plot the sine wave values for the example
plot_sine_wave_values(0, 12, 20)
