import random

# Flatten all points
X_cols = D[:, :3]
Y_cols = D[:, 3:]
X_all = X_cols.flatten()
Y_all = Y_cols.flatten()

points = np.vstack((X_all, Y_all)).T

def fit_line(p1, p2):
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = -(a*p1[0] + b*p1[1])
    return a, b, c

def distance(a, b, c, x, y):
    return abs(a*x + b*y + c) / np.sqrt(a*a + b*b)

def ransac(points, iterations=1000, threshold=0.2):
    best_inliers = []
    best_model = None
    
    for _ in range(iterations):
        sample = random.sample(list(points), 2)
        a, b, c = fit_line(sample[0], sample[1])
        
        inliers = []
        for p in points:
            if distance(a,b,c,p[0],p[1]) < threshold:
                inliers.append(p)
                
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (a,b,c)
    
    return best_model, np.array(best_inliers)

# Find 3 lines
remaining = points.copy()
models = []

for i in range(3):
    model, inliers = ransac(remaining)
    models.append(model)
    
    # Remove inliers
    remaining = np.array([p for p in remaining if p.tolist() not in inliers.tolist()])

# Plot
plt.scatter(points[:,0], points[:,1], s=5)

xx = np.linspace(min(X_all), max(X_all), 100)

for (a,b,c) in models:
    yy = (-a*xx - c)/b
    plt.plot(xx, yy)

plt.title("RANSAC - 3 Lines")
plt.show()
