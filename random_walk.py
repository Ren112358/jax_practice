import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of the plot using seaborn
sns.set(style='darkgrid')

# Parameter settings
num_steps = 1000  # Number of steps

# Function to simulate the random walk
def random_walk():
    # Define the possible movement directions: right, up, left, down
    directions = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]])  # Right, Up, Left, Down
    
    # Initialize the position at (0, 0)
    position = jnp.array([0, 0])  # Initial position at the origin (0, 0)
    positions = jnp.zeros((num_steps, 2), dtype=int)  # Array to record the positions
    positions = positions.at[0].set(position)  # Set the initial position at index 0
    
    # Simulate the movement for each step
    for i in range(1, num_steps):
        move = jax.random.choice(jax.random.PRNGKey(i), directions)  # Choose a random direction
        position = position + move  # Update the position by adding the move
        positions = positions.at[i].set(position)  # Record the new position
    
    return positions

# Run the random walk simulation
positions = random_walk()

# Plot the random walk trajectory
plt.plot(positions[:, 0], positions[:, 1], marker='o', markersize=3, linestyle='-')

# Highlight the initial position (0, 0) with a red dot
plt.scatter(0, 0, color='red', s=20, label="Initial Position", zorder=5)

# Set the title and labels for the axes
plt.title("Random Walk on a Grid", fontsize=14)
plt.xlabel("X Position", fontsize=12)
plt.ylabel("Y Position", fontsize=12)

# Show the plot
plt.show()

