import numpy as np

from riskmap import *
from riskmap.BallisticFall import GenerateImpactMap as generate_impact_map
from riskmap.BallisticFall import GenerateBallisticFall as generate_ballistic_fall
from riskmap.BallisticFall import BallisticFall as ballistic_fall


class BallisticRiskMap():
    def __init__(self, ground_map, aircraft, n_sigma, hdg_samples, num_falls, sim_time, v0, altitude_slices, hdg_sigma):
        self.hdg_samples = np.arange(0, np.pi * 2, 2 * np.pi / hdg_samples)
        self.ground_map = ground_map
        self.aircraft = aircraft
        self.n_sigma = n_sigma
        self.risk_map = dict()
        self.altitude_slices = altitude_slices
        for hdg in self.hdg_samples:
            if hdg not in self.risk_map:
                self.risk_map[np.round(hdg, 4)] = dict()

            impact_maps = generate_impact_map(num_falls, sim_time, v0, hdg, hdg_sigma, altitude_slices, aircraft)
            for impact_map in impact_maps:
                self.risk_map[np.round(hdg, 4)][np.round(impact_map.altitude, 0)] = impact_map.generate_probability_map(
                    n_sigma, ground_map)
        dummy_fall = generate_ballistic_fall(aircraft.c, aircraft.S, aircraft.m, 0., v0, sim_time)

        # Ballistic fall trajectory is solved numerically, and so it is sampled in time.
        # Too many samples are then above the same cell in the ground map
        # Reduce the number of samples by resampling in distance
        dist_samples = np.arange(0, dummy_fall.dist[-1], int(np.round(self.ground_map.resolution / 2, 0)))
        dist_samples = np.append(dist_samples, dummy_fall.dist[-1])
        idxs = [np.where(np.array(dummy_fall.dist) <= dist_sample)[0][-1] for dist_sample in dist_samples]

        dist = dummy_fall.dist[idxs]
        alt = dummy_fall.alt[idxs]
        v_h = dummy_fall.v_h[idxs]
        v_v = dummy_fall.v_v[idxs]
        self.dummy_fall = ballistic_fall(dist, alt, v_h, v_v, dummy_fall.init_hdg)

    """
    Method to determine a risk induced by potential ballistic failure along the given trajectory.
    Args:   
        Path path   # path to be assesed
    Returns:
        float risk  # risk induced by potential ballistic fall along the path
    """

    def get_risk(self, path):

        # TODO -- implement me

        if not path or not path.poses:
            # Handle missing or invalid data
            return float('inf')

        # Check if the density layer is valid
        density_layer = self.ground_map.layers['density']
        shelter_layer = self.ground_map.layers['density']
        height_layer = self.ground_map.layers['height']

        # Check for empty rows or columns
        if (density_layer.shape[0] == 0 or density_layer.shape[1] == 0 or shelter_layer.shape[0] == 0
                or shelter_layer.shape[1] == 0 or height_layer.shape[0] == 0 or height_layer.shape[1] == 0):
            # Return infinity if the layer is invalid
            return float('inf')

        total_risk = 0.0

        for pose in path.poses:
            # Extract position (x, y, z)
            position = pose.position
            x, y, altitude = position.x, position.y, position.z
            # Extract the yaw (heading angle) from the quaternion
            heading, _, _ = pose.orientation.to_Euler()
            # Normalize heading to [0, 2π]
            if heading < 0:
                heading += 2 * np.pi

            ground_height = 0.0

            # Iterate over the dummy fall's trajectory to find where it touches the ground
            for i in range(len(self.dummy_fall.dist)):
                # Calculate the current (x, y) position in ENU coordinates
                real_map_x = x + self.dummy_fall.dist[i] * np.cos(heading)
                real_map_y = y + self.dummy_fall.dist[i] * np.sin(heading)

                # Convert the (x, y) position to grid indices in the ground map
                grid_idx = self.ground_map.ENUtoGrid((real_map_x, real_map_y))

                # Check if the indices are within the bounds of the ground map
                if (grid_idx[0] <= self.ground_map.width and grid_idx[1] <= self.ground_map.height):

                    # Get the ground height at this grid cell
                    ground_height = height_layer[grid_idx[1], grid_idx[0]]

                    # Check if the UAV's altitude is less than or equal to the ground height
                    if self.dummy_fall.alt[i] + altitude <= ground_height:
                        break

            # Calculate the fallen altitude
            fallen_altitude = altitude - ground_height
            # Find the closest precomputed heading
            closest_heading = min(self.hdg_samples, key=lambda h: abs(h - heading))
            # Find the closest precomputed fallen altitude
            closest_altitude = min(self.altitude_slices, key=lambda alt: abs(alt + fallen_altitude))

            # Select the closest impact probability map
            impact_map = self.risk_map[np.round(closest_heading, 4)][np.round(closest_altitude, 0)]

            total_risk += self.calculate_risk(x, y, impact_map)

        return total_risk


    def calculate_risk(self, x, y, impact_map):

        # Initialize total risk
        total_risk = 0.0

        prob_map = impact_map.prob_map
        prob_map /= np.sum(prob_map)
        grid_x, grid_y = self.ground_map.ENUtoGrid((x, y))

        # Iterate over each cell in the probability map
        for deviation in range(prob_map.shape[0]):
            for distance in range(prob_map.shape[1]):
                grid_pos_x = grid_x + distance + impact_map.offset_idx[1]
                grid_pos_y = grid_y + deviation + impact_map.offset_idx[0]

                # Skip if prob_map points are out of ground map bounds
                if (grid_pos_x < 0 or grid_pos_y < 0 or grid_pos_y >= self.ground_map.height
                        or grid_pos_x >= self.ground_map.width):
                    return float('inf')

                density_at_cell = self.ground_map.layers['density'][grid_pos_y, grid_pos_x] / 1000000  # Population density
                prob_at_cell = prob_map[deviation, distance]  # Probability at this cell
                if prob_at_cell > 0 and density_at_cell > 0:
                    shelter_at_cell = self.ground_map.layers['shelter'][grid_pos_y, grid_pos_x]

                    # Impact energy (kinetic energy)
                    E = 0.5 * self.aircraft.m * (impact_map.impact_speed ** 2)

                    # Calculate M for this cell
                    risk_multiplier = self.M(density_at_cell, E, impact_map.impact_angle, shelter_at_cell)
                    # Add to total risk
                    # print("prob: ", prob_at_cell, "risk: ", risk_multiplier)
                    total_risk += prob_at_cell * risk_multiplier

        return total_risk


    def p_hit(self, dens_x, gama):
        A_exp = self.A_exp(gama)
        return dens_x * A_exp


    def A_exp(self, gama):
        r_p = 0.2
        r_uav = self.aircraft.r
        hp = 1.8
        gama = abs(gama)
        tan_gama = np.tan(gama)
        return 2 * (r_p + r_uav) * (hp / tan_gama) + np.pi * (r_p + r_uav) ** 2


    def M(self, dens_x, E, gama, shelter_at_cell):
        p_hit = self.p_hit(dens_x, gama)
        p_casaulty = self.p_casaulty(E, shelter_at_cell)
        return p_hit * p_casaulty


    def p_casaulty(self, E, shelter_at_cell):
        alpha = E
        beta = 34
        S_x = shelter_at_cell
        k = np.minimum(1, (beta / E) ** (3 / S_x))
        return (1 - k) / (1 - 2 * k + np.sqrt(alpha / beta) * ((beta / E) ** (3 / S_x)))
