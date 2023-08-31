import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import pandas as pd

# Define the ReLU function
def relu(x):
    return np.maximum(0, x)

# Define the x range for plotting (-10 to 10)
x = np.linspace(-10, 10, 100)

# Apply the ReLU function to each element in x
y = relu(x)

# Plot the ReLU function
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.title('ReLU Function')
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def cross_entropy_loss(p, q):
    """ Compute Cross Entropy Loss given true distribution p and estimated distribution q """
    return -np.sum(p * np.log(q) + (1-p) * np.log(1-q))

# Binary classification: true label is 1
p = np.array([0, 1])  # true distribution (one-hot encoded), in this case, class 1
estimated_prob = np.linspace(0.01, 0.99, 100)  # range of estimated probabilities for class 1
losses = [cross_entropy_loss(p, np.array([1-q, q])) for q in estimated_prob]

# Plotting
plt.plot(estimated_prob, losses, label='Cross-Entropy Loss')
plt.xlabel('Estimated Probability for Class 1')
plt.ylabel('Loss')
plt.title('Cross-Entropy Loss for Binary Classification (True Label = Class 1)')
plt.legend()
plt.grid()
plt.show()


def training_time():

    strategies = ["GEM", "Cumulative", "Generative Replay", "EWC", "Naive", "CWR*"]
    training_times = ['13h 48min', '10h 40min', '2h 59min', '2h 5min', '2h 1min', '1h 57min']
    # Convert training times to timedelta objects
    time_deltas = [timedelta(hours=int(t.split('h')[0]), minutes=int(t.split()[1].split('m')[0])) for t in training_times]

    # Convert timedelta objects to total hours for plotting
    training_hours = [td.total_seconds() / 3600 for td in time_deltas]

    # Create a DataFrame for plotting
    data = pd.DataFrame({'Labels': strategies, 'Training Time (hours)': training_hours})

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(data['Labels'], data['Training Time (hours)'], color=['blue', 'green', 'red'])
    plt.xlabel('Labels')
    plt.ylabel('Training Time (hours)')
    plt.title('Training Times for Different Strategies')
    # Adjust layout to prevent label cutoff
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.2)
    plt.show()

if __name__ == "__main__":
    training_time()