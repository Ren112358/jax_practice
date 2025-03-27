import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for better aesthetics
sns.set(style="darkgrid")

# Constants
g = 9.81  # Gravitational acceleration [m/s^2]
m = 1.0   # Mass [kg]
v0 = 20.0  # Initial velocity [m/s]
angle = 45.0  # Launch angle [degrees]
angle_rad = jnp.radians(angle)  # Convert launch angle to radians
y0 = 0.0   # Initial position [m]
x0 = 0.0   # Initial position [m]

# Initial velocity components
vx0 = v0 * jnp.cos(angle_rad)
vy0 = v0 * jnp.sin(angle_rad)

# Function to define the equations of motion (velocity and position updates)
def equations_of_motion(t, state):
    x, y, vx, vy = state
    dxdt = vx
    dydt = vy
    dvxdt = 0.0  # No air resistance
    dvydt = -g   # Gravitational acceleration
    return jnp.array([dxdt, dydt, dvxdt, dvydt])

# Initial state (position and velocity)
initial_state = jnp.array([x0, y0, vx0, vy0])

# Simulation parameters: maximum time and time step
t_max = 5.0  # Maximum time [seconds]
dt = 0.01    # Time step [seconds]
times = jnp.arange(0, t_max, dt)

# Runge-Kutta method to numerically solve the equations of motion
@jax.jit
def runge_kutta(t, state):
    # Perform the Runge-Kutta integration
    k1 = equations_of_motion(t, state)
    k2 = equations_of_motion(t + dt / 2, state + dt / 2 * k1)
    k3 = equations_of_motion(t + dt / 2, state + dt / 2 * k2)
    k4 = equations_of_motion(t + dt, state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Calculate state at each time step
states = []
state = initial_state
for t in times:
    states.append(state)
    state = runge_kutta(t, state)

states = jnp.array(states)

# Compute kinetic energy, potential energy, and total energy
x, y, vx, vy = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
kinetic_energy = 0.5 * m * (vx**2 + vy**2)
potential_energy = m * g * y
total_energy = kinetic_energy + potential_energy

# Plotting the results with Seaborn's style
plt.figure(figsize=(10, 6))

# Plot the projectile trajectory
plt.subplot(2, 1, 1)
plt.plot(x, y, label="Trajectory", color='b')
plt.xlabel("x [m]", fontsize=12)
plt.ylabel("y [m]", fontsize=12)
plt.title("Projectile Motion", fontsize=14)
plt.grid(True)

# Plot energy over time
plt.subplot(2, 1, 2)
plt.plot(times, kinetic_energy, label="Kinetic Energy", color='r')
plt.plot(times, potential_energy, label="Potential Energy", color='g')
plt.plot(times, total_energy, label="Total Energy", linestyle="--", color='k')
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("Energy [J]", fontsize=12)
plt.title("Energy vs Time", fontsize=14)
plt.legend()
plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

