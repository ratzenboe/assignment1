"""
Implementation of PreDeCon for course Datamining, winter semester 2017, University of Vienna.
All complex objects are numpy's ndarrays, if not otherwise specified.
"""

import numpy
import math


class PreDeCon:
    """
    Class implementing PreDeCon algorithm.
    Note: Currently all of the method stubs are intended to calculate values over the entire dataset. Additional methods
    may be restricted to specified subsets and the signatures of existing ones changed if deemed reasonable for
    performance or other reasons.
    """

    # Collection of data points' attributes.
    data = None

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

    def __init__(self,
                 data,
                 hyperparameters):
        """
        Generates new instance, copies data into class attributes.
        :param data: ndarray of ndarrays. Rows correspond to data points, columns to attribute values.
        :param hyperparameters: Dictionary with hyperparameters (pattern: "param_" + parameter name, e. g.
        "param_epsilon").
        """

        self.data = data
        self.param_mu = hyperparameters["mu"]
        self.param_epsilon = hyperparameters["epsilon"]
        self.param_theta = hyperparameters["theta"]
        self.param_lambda = hyperparameters["lambda"]
        self.param_kappa = hyperparameters["kappa"]

    def run(self):
        """
        Runs algorithm.
        :return: ndarray with cluster IDs for each point. None is a possible value (for points regarded as noise).
        """

        return None

    # ----------------------------
    # Auxiliary methods.
    # ----------------------------

    def calculate_epsilon_neighbourhoods(self):
        """
        Returns epsilon-neighbourhoods for all points.
        :return: ndarray of ndarrays, symmetric. Rows <-> origin points, columns <-> destination points; value: true
        if in each others epsilon neighbourhood (assuming symmetric distance function), otherwise false.
        """

        return None

    # ----------------------------
    # Definitions.
    # ----------------------------

    def calculate_variances_along_attributes(self):
        """
        Definition #1.
        :return: ndarray of ndarrays. Rows <-> points, columns <-> attribute variances (same sequence as in the supplied
        dataset).
        """

        return None

    def calculate_subspace_preference_dimensionalities(self):
        """
        Definition #2.
        :return: ndarray of integers. Values: SPD for point at index.
        """

        return None

    def calculate_preference_weighted_similarity_measures(self):
        """
        Definition #3.
        Note: Probably better to integrate actual calculation in calculate_general_preference_weighted_similarities()
        and deprecate this method.
        :return: ndarray of ndarrays. Rows <-> origin point, columns <-> destinations; value: PWS.
        """

        return None

    def calculate_general_preference_weighted_similarities(self):
        """
        Definition #4.
        :return: ndarray of ndarrays. Rows <-> origin points, columns <-> destinations; value: GPWS.
        """

        return None

    def calculate_preference_weighted_epsilon_neighbourhood(self):
        """
        Definition #5.
        :return: ndarray of ndarrays; symmetric. Rows <-> origin points, columns <-> destination points; value: true if in each
        other's PWE-neighbourhood, otherwise false.
        """

        return None

    def calculate_preference_weighted_core_points(self):
        """
        Definition 6.
        :return: ndarray. true if point at index is PWCP, otherwise false.
        """

        return None

    def calculate_direct_preference_weighted_reachablilities(self):
        """
        Definition 7.
        :return: ndarray of ndarrays; symmetric. Rows <-> origin points, columns <-> destination points; value: true if
        DPW-reachable, otherwise false.
        """

        return None

    def calculate_preference_weighted_reachablilities(self):
        """
        Definition 8.
        :return:  ndarray of ndarrays; symmetric. Rows <-> origin points, columns <-> destination points; value: true if
        PW-reachable, otherwise false.
        """

        return None

    def calculate_preference_weighted_connectivities(self):
        """
        Definition 9.
        :return: ndarray of ndarrays; symmetric. Rows <-> origin points, columns <-> destination points; value: true if
        PW-connected, otherwise false.
        """

        return None

    def calculate_subspace_preference_cluster(self):
        """
        Definition 10.
        :return: ndarray of integers. Values: Cluster ID of point with corresponding index.
        """

        return None
