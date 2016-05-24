import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

# Generate data:
lambda_l2 = [0.001, 0.01, 0.1, 1, 10]
embed_dim = [64, 128, 256]
x = []
y = []
for l2 in lambda_l2:
    for dim in embed_dim:
        x.append(l2)
        y.append(dim)
# x, y, z = 10 * np.random.random((3,10))
z = [0.838, 0.854, 0.846, 0.822, 0.852, 0.848, 0.838, 0.844, 0.848, 0.808, 0.798, 0.822, 0.656, 0.682, 0.756]

print x
print y
print z


# Set up a regular grid of interpolation points
xi, yi = np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
zi = rbf(xi, yi)

fig = plt.figure()
fig.suptitle('accuracy (10 epochs)', fontsize=20)
plt.xlabel('lambda_l2_regularization', fontsize=18)
plt.ylabel('embedding_dimension', fontsize=16)
plt.imshow(zi, vmin=min(z), vmax=max(z), origin='lower',
           extent=[min(x), max(x), min(y), max(y)], aspect="auto")
plt.scatter(x, y, c=z)
plt.colorbar()
plt.show()
