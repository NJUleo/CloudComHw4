import numpy as np

# Sample data for r, y, u
r = np.array([0.2, 0.4, 0.3, 0.2])  # Replace with actual values
y = np.array([0, 0.2, 0.4, 0.3])  # Replace with actual values
u = np.array([1, 3, 2, 1])  # Replace with actual values

# Ensure the vectors have the same length
assert len(r) == len(y) == len(u)

# Stack y and u horizontally to create a matrix
X = np.column_stack((y, u))

# Add a column of ones for the intercept term
X = np.column_stack((X, np.ones(len(r))))

# Perform linear regression using least squares
# Here, r is the dependent variable and X is the independent variable
coefficients = np.linalg.lstsq(X, r, rcond=None)[0]

# coefficients[0] is 'a' and coefficients[1] is 'b'
a, b = coefficients[:2]

print("Coefficient a:", a)
print("Coefficient b:", b)
