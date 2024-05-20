import numpy as np
import matplotlib.pyplot as plt

# Define the Gaussian function
def gaussian(x, mu, sigma):
    return np.exp(-1 * ((x - mu) / sigma) ** 2) / (2.0 * sigma * np.sqrt(2 * np.pi))

# Define the parameters
ground_beat = 1.5  # Example ground beat in seconds
sigma = 0.04  # Standard deviation of 40ms

# Generate x values
x_values = np.linspace(ground_beat - 0.1, ground_beat + 0.1, 100)  # Adjust the range as needed

# Calculate y values using the Gaussian function
y_values = gaussian(x_values, ground_beat, sigma)

# Plot the Gaussian function
plt.plot(x_values, y_values, color='blue', label='Gaussian Function')
plt.xlabel('Time (s)')
plt.ylabel('Gaussian Value')
plt.title('Gaussian Function with Standard Deviation 40ms')
plt.legend()
plt.grid(True)
plt.show()
