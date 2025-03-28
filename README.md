# JAX Practice Repository
This project contains various Python scripts to practice and experiment with JAX. The focus is on building knowledge in numerical methods, scientific computing and machine learning applications.

# Installation
To set up the environment and install the required dependencies, follow these steps:
1. Clone this repository:
   ```bash
   git clone https://github.com/<your_username>/jax_practice.git
   cd jax_practice
   ```

2. Install dependencies using Pipenv:
   ```bash
    pipenv install
   ```

# Running the Code
After the environment is set up, you can run any of the practice scripts by using the following command:

```bash
pipenv run python <file_name>.py
```

For example, to run the `projectile_motion_energy.py` script:

```bash
pipenv run python projectile_motion_energy.py
```

# Simulations in this Repository
## 1. Projectile Motion and Energy Conservation
* file name: `projectile_motion_energy.py`
* description: This simulation models the motion of a projectile under gravity. It calculates and visualizes the projectile's trajectory while also computing and displaying the kinetic energy, potential energy and total energy over calculation time. This example helps demonstrate the principle of energy conservatin in projectile motion.

## 2. Random Walk
* file name: `random_walk.py`
* description: A random walk is a mathematical process where an object moves in random directions at each step. In a 2D grid, it typically moves up, down, left, or right, with each direction chosen randomly. The path of the object is recorded, creating a random trajectory.


# License
[MIT License](LICENSE)

# References
* [JAX](https://docs.jax.dev/en/latest/index.html)
