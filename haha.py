import matplotlib.pyplot as plt

# Given list of sample points
sample = [(1, 2), (3, 4)]

# Unpacking the points into x and y coordinates
x, y = zip(*sample)

# Plotting the points
plt.scatter(x, y)
plt.title("Sample Data Points")
plt.xlabel("x values")
plt.ylabel("y values")
plt.grid(True)
plt.show()
