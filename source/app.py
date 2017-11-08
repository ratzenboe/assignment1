"""
Wrapper script for executing PreDeCon on dataset.
"""

from PreDeCon import PreDeCon
import numpy as np
import sys

data = np.array([[1, 6], [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [7, 8], [7, 7], [7, 6], [7, 5], [7, 4], [7, 3]])
hyperparameters = dict()
hyperparameters["mu"] = 3
hyperparameters["epsilon"] = 1
hyperparameters["theta"] = 0.25
hyperparameters["lambda"] = 1
hyperparameters["kappa"] = 100

# Initiate algorithm.
predecon = PreDeCon(data=data, hyperparameters=hyperparameters)
# Run algorithm.
clusters = predecon.run()
print(clusters)
predecon.plot_results()
