"""
Wrapper script for executing PreDeCon on dataset.
"""

from PreDeCon import PreDeCon
from sklearn import metrics
import pandas as pd
import sys
import operator
from datetime import datetime

#data = np.array([[1, 6], [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [7, 8], [7, 7], [7, 6], [7, 5], [7, 4], [7, 3]])

elki_data = pd.read_csv("elki_data.csv", delimiter=" ", header=None, names=['x','y','cluster'])

# refine elki dataset
for i in range(0, len(elki_data["cluster"])):
    c = 0
    if elki_data["cluster"][i] == "Cluster1":
        c = 1
    elif elki_data["cluster"][i] == "Cluster2":
        c = 2
    else:
        c = 3
    elki_data["cluster"][i] = c

elki2_data = elki_data.drop(elki_data.columns[2], axis=1).values

hyperparameters = dict()
# min points
#hyperparameters["mu"] = 10
# max distance between points
#hyperparameters["epsilon"] = 0.3
# max variance (delta)
#hyperparameters["theta"] = 0.1
# max preference dimensionality of eps-neighborhood
#hyperparameters["lambda"] = 2
# weight factor
#hyperparameters["kappa"] = 20

# Initiate algorithm.
#predecon = PreDeCon(data=elki_data, hyperparameters=hyperparameters)
# Run algorithm.
#clusters = predecon.run()
#print(clusters)
#predecon.plot_exercise3_results()
#result_plot = predecon.plot_results()
# Save plot to disk
#result_plot.savefig("results.png")

mu = [2,6,10]
epsilon = [0.1, 0.3, 0.9]
theta = [0.1, 0.3, 0.5]
#lambda = [1,2]
kappa = [10,20,50]

vals = dict()
labels_true = elki_data["cluster"]
model = 0
ami = None
labels_pred = None
for m in mu:
    for e in epsilon:
        for t in theta:
            #for l in lambda:
            for k in kappa:
                hyperparameters["mu"] = m
                hyperparameters["epsilon"] = e
                hyperparameters["theta"] = t
                hyperparameters["lambda"] = 2
                hyperparameters["kappa"] = k
                predecon = PreDeCon(data=elki2_data, hyperparameters = hyperparameters)
                start = datetime.now()
                clusters, labels_pred = predecon.run()
                stop = datetime.now()
                ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
                homogeneity = metrics.homogeneity_score(labels_true, labels_pred)
                completeness = metrics.completeness_score(labels_true, labels_pred)
                v_score = metrics.v_measure_score(labels_true, labels_pred)
                vals[model] = {"mu":m,"epsilon":e,"theta": t, "lambda": 2, "kappa":k, "ami":ami, "time":stop-start, "predecon": predecon, "v_score": v_score, "completeness": completeness, "homogeneity": homogeneity}
                model = model + 1

final_v_score = max(vals.values(), key=operator.itemgetter('v_score'))
print("V_SCORE WINNER")
print(final_v_score)

res_v_score = final_v_score["predecon"].plot_results()

final_ami = max(vals.values(), key=operator.itemgetter('ami'))
print("AMI WINNER")
print(final_ami)

res_ami = final_ami["predecon"].plot_results()
