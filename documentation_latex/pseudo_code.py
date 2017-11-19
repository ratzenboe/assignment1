
# Determine necessary properties.
self.calculate_epsilon_neighbourhoods()
self.calculate_variances_along_attributes()
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
                        if (self.subspace_preference_dimensionalities[x_point]
                            <= self.param_lambda) and \
                            (x_point in self.preference_weighted_neighbourhoods[q_point]) and \
                            (not PreDeCon.is_classified(x_point, self.clusters)):
                            # only insert points which are not classified yet
                            direct_preference_reachable_from_point_q.append(x_point)
                else:
                    # A point is always reachable from it self
                    direct_preference_reachable_from_point_q.append(q_point)
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
