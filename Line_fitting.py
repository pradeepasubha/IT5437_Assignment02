import numpy as np
import matplotlib.pyplot as plt

# Load data
D = np.genfromtxt(r"C:\Users\Administrator\Desktop\UOM\Sem3\Computer Vision - Dr Ranga\Assignment 02\lines.csv",
                  delimiter=",", skip_header=1)

# First line data
x = D[:, 0]
y = D[:, 3]

# Stack into (N, 2) matrix
X = np.vstack((x, y)).T

# Compute centroid
centroid = np.mean(X, axis=0)

# Subtract mean
X_centered = X - centroid

# SVD
U, S, Vt = np.linalg.svd(X_centered)

# Line direction and normal
direction = Vt[0]
normal = Vt[-1]  # last row = normal to the line

# Line equation: ax + by + c = 0
a, b = normal
c = -a * centroid[0] - b * centroid[1]

print(f"Line parameters: {a:.4f}x + {b:.4f}y + {c:.4f} = 0")
slope = -a / b
intercept = -c / b
print(f"Slope:     {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"Equation:  y = {slope:.4f}x + ({intercept:.4f})")

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(x, y, label='Data points (line 1)', color='steelblue', s=30)
xx = np.linspace(min(x), max(x), 100)
yy = (-a * xx - c) / b
plt.plot(xx, yy, 'r', linewidth=2, label=f'TLS: y={slope:.4f}x+({intercept:.4f})')
plt.title("TLS Line Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
