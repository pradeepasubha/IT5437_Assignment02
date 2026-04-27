import numpy as np
import matplotlib.pyplot as plt

# Load data
D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)

# First line data
x = D[:, 0]
y = D[:, 3]

# Stack
X = np.vstack((x, y)).T

# Compute centroid
centroid = np.mean(X, axis=0)

# Subtract mean
X_centered = X - centroid

# SVD
U, S, Vt = np.linalg.svd(X_centered)

# Line direction
direction = Vt[0]
normal = Vt[1]

# Line equation: ax + by + c = 0
a, b = normal
c = -a*centroid[0] - b*centroid[1]

print(f"Line parameters: {a:.4f}x + {b:.4f}y + {c:.4f} = 0")

# Plot
plt.scatter(x, y)
xx = np.linspace(min(x), max(x), 100)
yy = (-a*xx - c)/b
plt.plot(xx, yy, 'r')
plt.title("TLS Line Fit")
plt.show()
