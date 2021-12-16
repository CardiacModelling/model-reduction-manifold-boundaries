import numpy as np
from MMR import r, rfull, rfull_reduced, j, Avv
from geodesic import geodesic, InitialVelocity
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# Choose starting parameters
x = np.log([1.0, 1.0])
v = InitialVelocity(x, j, Avv)

# Callback function used to monitor the geodesic after each step
def callback(geo):
    # Integrate until the norm of the velocity has grown by a factor of 10
    # and print out some diagnotistic along the way
    print("Iteration: %i, tau: %f, |v| = %f" %(len(geo.vs), geo.ts[-1], np.linalg.norm(geo.vs[-1])))
    return np.linalg.norm(geo.vs[-1]) < 100.0

# Construct the geodesic
# It is usually not necessary to be very accurate here, so we set small tolerances
geo_forward = geodesic(r, j, Avv, 2, 2, x, v, atol = 1e-2, rtol = 1e-2, callback = callback)  

# Integrate
geo_forward.integrate(25.0)

# plot the geodesic path to find the limit
# This should show the singularity at the "fold line" x[0] = x[1]
plt.figure(figsize=(4, 4), dpi=200)
plt.plot(geo_forward.ts, geo_forward.xs)
plt.xlabel(r'$\tau$')
plt.ylabel('Parameter values')
plt.legend([r'$\rho_1$', r'$\rho_2$'])
plt.tight_layout()
plt.savefig('Toy_model_parameter_values_geodesic_path.png')
# plt.show()

## Now construct contour plots in parameter space and model manifold in data space

r0 = r([0.0, 0.0])
xs = np.linspace(-5, 5, 101)
X = np.empty((101, 101))
Y = np.empty((101, 101))
Z = np.empty((101, 101))
C = np.empty((101, 101))
for i, x in enumerate(xs):
    for j, y in enumerate(xs):
        temp = r([x, y])
        X[j, i], Y[j, i], Z[j, i] = temp
        C[j, i] = np.linalg.norm(temp - r0)**2
        
# Plot geodesic path in parameter space with cost contours
plt.figure(figsize=(5, 4), dpi=200)
plt.contourf(xs, xs, C, 50, cmap=cm.viridis_r)
plt.plot(geo_forward.xs[:,0], geo_forward.xs[:,1], "r-")
plt.plot([0], [0], "ro")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel(r'$\log(\rho_1)$')
plt.ylabel(r'$\log(\rho_2)$')
plt.tight_layout()
plt.savefig('Toy_model_geodesic_path_parameter_space.png')
# plt.show()

N = C/C.max()

## Plot surface / geodesic in data space

fig = plt.figure(figsize=(8, 6), dpi=200)
ax = plt.subplot(projection='3d')
surf = ax.plot_surface(X, Y, Z, facecolors=cm.viridis_r(N),
                       linewidth=0, antialiased=False,
                       rstride=1, cstride=1, shade=False,
                       zorder=-1)

geo = geo_forward
X = np.empty(len(geo.xs))
Y = np.empty(len(geo.xs))
Z = np.empty(len(geo.xs))
for i, x in enumerate(geo.xs):
    X[i], Y[i], Z[i] = r(x)
ax.plot(X, Y, Z, color = (1,0,0), zorder=1000)

# Plot starting point of geodesic as a red dot
ax.plot([r0[0]], [r0[1]], [r0[2]], marker='o', c="r", markersize=5, zorder=1000)
ax.set_xlabel('Observation 1')
ax.set_ylabel('Observation 2')
ax.set_zlabel('Observation 3')
plt.tight_layout()
plt.savefig('Toy_model_geodesic_path_model_manifold.png')
# plt.show()

x0 = np.log([1.0, 1.0])
ts = np.array([0.0, 1.0, 2.0, 5.0])  # Time points to sample model.  We do not observe t = 0, but is necessary for the ODE solver
tfulls = np.linspace(0, 7, 71)

a, b = geo_forward.xs[-1]
new = a - b

plt.figure(figsize=(4, 4), dpi=200)
plt.plot(tfulls[1:], rfull(x0), label='Full model')
plt.plot(tfulls[1:], rfull_reduced([new, 0.0]), label='Reduced model')
for i, t in enumerate(ts[1:]):
    plt.axvline(t, linestyle='dashed', color='silver', label='System observations' if i == 0 else '')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.tight_layout()
plt.savefig('Toy_model_reduction.png')
# plt.show()

