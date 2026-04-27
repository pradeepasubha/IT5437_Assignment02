import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Load data
# ============================================================
D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)
X_cols = D[:, :3]   # columns x1, x2, x3
Y_cols = D[:, 3:]   # columns y1, y2, y3

# ============================================================
# Helper functions
# ============================================================

def tls_line(pts):
    """
    Fit a line to 2D points using Total Least Squares (TLS).
    Centers the data, applies SVD, and returns (a, b, c)
    for the line equation: a*x + b*y = c
    """
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    Z = pts - np.array([cx, cy])
    _, _, Vt = np.linalg.svd(Z)
    a, b = Vt[-1]          # normal vector = last right-singular vector
    c = a * cx + b * cy
    return a, b, c


def point_line_dist(pts, a, b, c):
    """
    Perpendicular (orthogonal) distance from each point to line a*x + b*y = c.
    """
    return np.abs(a * pts[:, 0] + b * pts[:, 1] - c) / np.sqrt(a**2 + b**2)


def ransac_line(pts, n_iter=2000, threshold=0.3):
    """
    RANSAC line fitting using TLS for model estimation.

    Parameters
    ----------
    pts       : (N, 2) array of 2D points
    n_iter    : number of RANSAC iterations
    threshold : inlier distance threshold

    Returns
    -------
    a, b, c   : TLS-fitted line parameters for best consensus set
    inliers   : indices of inlier points
    """
    np.random.seed(42)
    best_inliers = None
    best_count = 0

    for _ in range(n_iter):
        # Randomly sample 2 points
        idx = np.random.choice(len(pts), 2, replace=False)
        sample = pts[idx]
        if np.allclose(sample[0], sample[1]):
            continue

        # Fit candidate line
        a, b, c = tls_line(sample)

        # Count inliers
        dists = point_line_dist(pts, a, b, c)
        inliers = np.where(dists < threshold)[0]

        if len(inliers) > best_count:
            best_count = len(inliers)
            best_inliers = inliers

    # Refit using all inliers of the best consensus set
    a, b, c = tls_line(pts[best_inliers])
    return a, b, c, best_inliers


# ============================================================
# Part (a): TLS on first line only (x1, y1)
# ============================================================
x1 = X_cols[:, 0]
y1 = Y_cols[:, 0]

xm, ym = x1.mean(), y1.mean()
X = np.stack([x1 - xm, y1 - ym], axis=1)
U, S, Vt = np.linalg.svd(X)
a, b = Vt[-1]
c = a * xm + b * ym

slope_tls = -a / b
intercept_tls = c / b

print("=== Part (a): TLS on first line (x1, y1) ===")
print(f"Normal form:   {a:.6f}*x + {b:.6f}*y = {c:.6f}")
print(f"Slope (m):     {slope_tls:.6f}")
print(f"Intercept (b): {intercept_tls:.6f}")
print(f"Equation:      y = {slope_tls:.4f}x + ({intercept_tls:.4f})")

# ============================================================
# Part (b): RANSAC on all points (x1,x2,x3 / y1,y2,y3 flattened)
# ============================================================
X_all = X_cols.flatten()
Y_all = Y_cols.flatten()
points = np.stack([X_all, Y_all], axis=1)
N = len(points)

print("\n=== Part (b): RANSAC on all points ===")
remaining = np.ones(N, dtype=bool)
assignments = -np.ones(N, dtype=int)
line_params = []

for i in range(3):
    pts = points[remaining]
    a, b, c, local_inliers = ransac_line(pts, n_iter=2000, threshold=0.3)

    # Map local inlier indices back to global index space
    global_idx = np.where(remaining)[0]
    inlier_global = global_idx[local_inliers]
    assignments[inlier_global] = i
    remaining[inlier_global] = False

    slope = -a / b
    intercept = c / b
    line_params.append((a, b, c, slope, intercept))

    print(f"\nLine {i+1}:")
    print(f"  Normal form: {a:.6f}*x + {b:.6f}*y = {c:.6f}")
    print(f"  Slope:       {slope:.6f}")
    print(f"  Intercept:   {intercept:.6f}")
    print(f"  Equation:    y = {slope:.4f}x + ({intercept:.4f})")
    print(f"  Inliers:     {len(local_inliers)}")

# ============================================================
# Plotting
# ============================================================
colors = ['#E63946', '#2A9D8F', '#F4A261']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left plot: Part (a) ---
ax = axes[0]
ax.scatter(x1, y1, color='steelblue', s=30, zorder=3, label='Data points (line 1)')
xr = np.linspace(x1.min() - 0.5, x1.max() + 0.5, 200)
ax.plot(xr, slope_tls * xr + intercept_tls, 'r-', linewidth=2,
        label=f'TLS: y={slope_tls:.4f}x+({intercept_tls:.4f})')
ax.set_title('(a) TLS Fit — Line 1 (x1, y1)', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Right plot: Part (b) ---
ax = axes[1]
for i, (a, b, c, slope, intercept) in enumerate(line_params):
    mask = assignments == i
    ax.scatter(points[mask, 0], points[mask, 1], color=colors[i], s=25, zorder=3,
               label=f'Line {i+1} points')
    xr = np.linspace(points[mask, 0].min() - 0.5, points[mask, 0].max() + 0.5, 200)
    ax.plot(xr, slope * xr + intercept, color=colors[i], linewidth=2, linestyle='--',
            label=f'Line {i+1}: y={slope:.3f}x+({intercept:.3f})')

ax.set_title('(b) RANSAC Fits — All Three Lines', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('line_fitting.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as line_fitting.png")
