"""
Implementation of PreDeCon for course Datamining, winter semester 2017, University of Vienna.
All complex objects are numpy's ndarrays, if not otherwise specified.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque


class PreDeCon:
    """
    Class implementing PreDeCon algorithm.
    Note: Currently all of the method stubs are intended to calculate values over the entire dataset. Additional methods
    may be restricted to specified subsets and the signatures of existing ones changed if deemed reasonable for
    performance or other reasons.
    """

    # Collection of data points' attributes.
    data = None
    # Number of points.
    num_points = None
    # Number of dimensions.
    num_dimensions = None

    # Hyperparameter mu: Minimal number of points in epsilon-neighbourhood.
    param_mu = 0
    # Hyperparameter epsilon: Distance for neighbourhood.
    param_epsilon = 0
    # Hyperparameter theta: Variance threshold for subspace preference clusters.
    param_theta = 0
    # Hyperparameter lambda: Dimensionality threshold.
    param_lambda = 0
    # Hyperparameter kappa: Weight for subspace preference vectors.
    param_kappa = 0

    # Results.
    epsilon_neighbourhoods = None
    attribute_variances = None
    subspace_preference_vectors = None
    preference_weighted_neighbourhoods = None
    subspace_preference_dimensionalities = None
    pref_weighted_similarity_measures = None
    core_points = None
    clusters = dict()
    clusters["noise"] = []

    def __init__(self,
                 data,
                 hyperparameters):
        """
        Generates new instance, copies data into class attributes.
        :param data: ndarray of ndarrays. Rows correspond to data points, columns to attribute values.
        :param hyperparameters: Dictionary with hyperparameters (pattern: "param_" + parameter name, e. g.
        "param_epsilon").
        """

        mpl.style.use('ggplot')

        # Parameters and hyperparameters.
        self.data = data
        self.param_mu = hyperparameters["mu"]
        self.param_epsilon = hyperparameters["epsilon"]
        self.param_theta = hyperparameters["theta"]
        self.param_lambda = hyperparameters["lambda"]
        self.param_kappa = hyperparameters["kappa"]

        self.num_points = len(data)
        self.num_dimensions = len(data[0])

    def run(self, verbose=False):
        """
        Runs algorithm.
        :param verbose: Determines if results should be printed.
        :return: Dictionary with cluster ID -> list of data points in this cluster.
        """

        # Determine necessary properties.
        self.calculate_epsilon_neighbourhoods()
        self.calculate_variances_along_attributes(verbose=True)
        self.calculate_subspace_preference_dimensionalities()
        self.calculate_subspace_preference_vectors()
        self.calculate_preference_weighted_similarity_measures()
        self.calculate_preference_weighted_epsilon_neighbourhood()
        self.calculate_preference_weighted_core_points()

        # Use properties in order to assign clusters.
        cluster_id = 0

        # Loop over all points
        for point in range(0, self.num_points):
            if not PreDeCon.is_classified(point, self.clusters):
                if point in self.core_points:
                    core_point_i = point
                    # Expand new cluster
                    cluster_id += 1
                    self.clusters[cluster_id] = []
                    queue = deque(self.preference_weighted_neighbourhoods[core_point_i])
                    # loop through points in preference_weighted_neighbourhood
                    # of the core_point_i
                    while queue:
                        q_point = queue.popleft()
                        # Get all points which are direct preference reachable
                        # from point q: Definition 8:
                        # DIRREACH(q,x) <=> q is core point,
                        #                   PDIM(N_e(x)) <= lambda,
                        #                   x is in preference weighted neighbourhood of q
                        direct_preference_reachable_from_point_q = []
                        if q_point in self.core_points:
                            for x_point in range(0, self.num_points):
                                if (self.subspace_preference_dimensionalities[x_point] <= self.param_lambda) and \
                                        (x_point in self.preference_weighted_neighbourhoods[q_point]) and \
                                        (not PreDeCon.is_classified(x_point, self.clusters)):
                                    # only insert points which are not classified yet
                                    direct_preference_reachable_from_point_q.append(x_point)

                        # Loop over direct preference reachable points
                        # and assign unclassified points to the cluster
                        for reachable_point in direct_preference_reachable_from_point_q:
                            if not PreDeCon.is_classified(reachable_point, self.clusters):
                                queue.append(reachable_point)
                            if (not PreDeCon.is_classified(reachable_point, self.clusters)) or \
                                    (reachable_point in self.clusters["noise"]):
                                self.clusters[cluster_id].append(reachable_point)
                                # points are uniquely assigned
                                if reachable_point in self.clusters["noise"]:
                                    self.clusters["noise"].remove(reachable_point)
                else:
                    self.clusters["noise"].append(point)

        if verbose:
            print(self.clusters)

        return self.clusters

    # ----------------------------
    # Auxiliary methods.
    # ----------------------------

    def calculate_epsilon_neighbourhoods(self, verbose=False):
        """
        Returns epsilon-neighbourhoods for all points.
        :param verbose: Determines if results should be printed.
        :return: List of lists with epsilon neighbourhoods for each point.
        """

        self.epsilon_neighbourhoods = []
        for i in range(0, self.num_points):
            epsilon_neighbour_indices = [j for j, point_j in enumerate(self.data)
                                         if (np.linalg.norm(self.data[i] - point_j) <= self.param_epsilon)]
            self.epsilon_neighbourhoods.append(epsilon_neighbour_indices)

        if verbose:
            for i, point in enumerate(self.epsilon_neighbourhoods):
                print("P" + str(i + 1), ["P" + str(i + 1) for i in point])

        return self.epsilon_neighbourhoods

    def calculate_subspace_preference_vectors(self, verbose=False):
        """
        :param verbose:
        :return: List of subspace preference vectors (as lists) for each data point.
        """

        self.subspace_preference_vectors = [[] for i in range(0, self.num_points)]
        for i_datapoint in range(0, self.num_points):
            self.subspace_preference_vectors[i_datapoint] = \
                [1 if att_var > self.param_theta else self.param_kappa
                 for att_var in self.attribute_variances[i_datapoint]]

        if verbose:
            for i, point in enumerate(self.subspace_preference_vectors):
                print("P" + str(i + 1), [i for i in point])

        return self.subspace_preference_vectors

    def plot_results(self):
        fig, ax = plt.subplots()
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        ax.set_ylim(0, 12)
        ax.set_xlim(0, 12)
        ax.xaxis.set_ticks([i for i in range(0, 13)])
        ax.yaxis.set_ticks([i for i in range(0, 13)])
        ax.grid(True)
        for i in range(self.data.shape[0]):
            ax.annotate("p" + str(i + 1), (self.data[i, 0], self.data[i, 1]), xytext=(self.data[i, 0], self.data[i, 1] + 0.5))

        colors = ["blue", "yellow", "green", "black"]
        coloring = []
        for point in range(0, self.num_points):
            if point in self.clusters["noise"]:
                coloring.append(colors[0])
            else:
                for cluster_i in range(1, len(self.clusters)):
                    if point in self.clusters[cluster_i]:
                        coloring.append(colors[cluster_i])

        color_patches = [mpl.patches.Patch(color="yellow", label="Cluster 1"),
                         # mpl.patches.Patch(color = "green", label = "Cluster 2"),
                         mpl.patches.Patch(color="blue", label="Noise")]
        ax.legend(handles=color_patches, loc=9)
        plt.scatter(self.data[:, 0], self.data[:, 1], c=coloring);

        plt.show()

    @staticmethod
    def is_classified(point, clusters):
        for cluster in clusters.values():
            if point in cluster:
                return True
        return False

    # ----------------------------
    # Definitions.
    # ----------------------------

    def calculate_variances_along_attributes(self, verbose=False):
        """
        Definition #1.
        :param verbose: Determines if results should be printed.
        :return: List of lists with variances for each attribute (columns) for each data point (rows).
        """

        self.attribute_variances = [[] for i in range(0, self.num_points)]
        index = 0
        for i_datapoint in range(0, self.num_points):
            number_of_epsilon_neighbours = len(self.epsilon_neighbourhoods[i_datapoint])
            difference = np.subtract(self.data[i_datapoint], self.data[self.epsilon_neighbourhoods[i_datapoint]])
            for i_attribute in range(0, self.num_dimensions):
                attribute_var = np.divide(np.sum(np.square(difference[:, i_attribute])),
                                          number_of_epsilon_neighbours)
                self.attribute_variances[i_datapoint].append(attribute_var)

        if verbose:
            for i, point in enumerate(self.attribute_variances):
                print("P" + str(i + 1), [i for i in point])

        return self.attribute_variances

    def calculate_subspace_preference_dimensionalities(self, verbose=False):
        """
        Definition #2.
        :param verbose: Determines if results should be printed.
        :return: List of scalar dimensionality values for each datapoint.
        """

        self.subspace_preference_dimensionalities = []

        for i_datapoint in range(0, self.num_points):
            self.subspace_preference_dimensionalities.append(
                np.sum(att_var <= self.param_theta for att_var in self.attribute_variances[i_datapoint])
            )

        if verbose:
            print("Subspace preference dimensionalities: ")
            for i, point in enumerate(self.subspace_preference_dimensionalities):
                print("P" + str(i + 1), point)

        return self.subspace_preference_dimensionalities

    def calculate_preference_weighted_similarity_measures(self, verbose=False):
        """
        Definition #3.
        I think the weights should be calculated as 1/subspace_preference_vectors as it is stated in the slides and mentioned
        in the paper: "The epsilon-neighborhood of a 2-dimensional point p exhibits low variance along attribute A1 and
        high variance along attribute A2 . The similarity measure dist_p weights attributes with low variance considerably
        lower (by the factor κ) than attributes with a high variance."
        :param verbose: Determines if results should be printed.
        :return: List of PWSMs (as lists) from each point to to all other points.
        """

        self.pref_weighted_similarity_measures = []
        subspace_preference_vectors_array = np.array(self.subspace_preference_vectors)
        for i_datapoint in range(0, self.num_points):
            self.pref_weighted_similarity_measures.append(
                np.sqrt(
                    np.sum(
                        np.multiply(
                            np.divide(1,
                                      subspace_preference_vectors_array[i_datapoint]
                                      ),
                            np.square(
                                np.subtract(self.data[i_datapoint], self.data)
                            )
                        ), axis=1
                    )
                )
            )

        if verbose:
            pref_weighted_similarity_measures = np.reshape(self.pref_weighted_similarity_measures,
                                                           (-1, self.num_points))
            for i, point in enumerate(pref_weighted_similarity_measures):
                print("P" + str(i + 1) + "\n{}".format(point))

        return self.pref_weighted_similarity_measures

    def calculate_preference_weighted_epsilon_neighbourhood(self, verbose=False):
        """
        Definition #5.
        :param verbose: Determines if results should be printed.
        :return: List of neighbours (as list) for each datapoint.
        """

        self.preference_weighted_neighbourhoods = [[] for i in range(0, self.num_points)]

        for i_datapoint in range(0, self.num_points):
            for j_datapoint in range(0, self.num_points):
                dist_pref = max(
                    self.pref_weighted_similarity_measures[i_datapoint][j_datapoint],
                    self.pref_weighted_similarity_measures[j_datapoint][i_datapoint]
                )

                if dist_pref <= self.param_epsilon:
                    self.preference_weighted_neighbourhoods[i_datapoint].append(j_datapoint)

        if verbose:
            for i, point in enumerate(self.preference_weighted_neighbourhoods):
                print("P" + str(i + 1), ["P" + str(i + 1) for i in point])

        return self.preference_weighted_neighbourhoods

    def calculate_preference_weighted_core_points(self, verbose=False):
        """
        Definition 6.
        :param verbose: Determines if results should be printed.
        :return: List of core point indices.
        """

        self.core_points = []
        for i_datapoint in range(0, self.num_points):
            if self.subspace_preference_dimensionalities[i_datapoint] <= self.param_lambda and \
                            len(self.preference_weighted_neighbourhoods[i_datapoint]) >= self.param_mu:
                self.core_points.append(i_datapoint)

        if verbose:
            print("Resulting Core Points: \n{} \n{}".format(
                self.core_points,["P" + str(p + 1) for p in self.core_points]))

        return self.core_points
