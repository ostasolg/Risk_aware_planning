import numpy as np
import scipy.integrate as integrate
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from riskmap import GroundMap

"""
A ballistic fall class. The ballistic trajectory is sampled according to the ODE solver
and values at each sample are known.
Args:
    dist    # Distance from origin of the samples
    alt     # Altitude difference w.r.t. origin of the samples
    v_h     # Horizontal speed of the samples
    v_v     # Vertical speed of the samples
    hdg     # Initial heading of the ballistic fall
"""
class BallisticFall():
    def __init__(self, dist, alt, v_h, v_v, hdg):
        self.dist = dist
        self.alt = alt
        self.v_h = v_h
        self.v_v = v_v
        self.init_hdg = hdg

"""
Implementation of impact probability map.
Args:
    prob_map        # 2D matrix with probability of impact in the given cell
    offset          # Offset of prob_map origin w.r.t. global origin in world coordinates
    offset_idx      # Offset of prob_map origin w.r.t. global origin in ground map coordinates
    impact_speed    # Predicted aircraft impact speed for the given impact probability map
    impact_angle    # Predicted aircraft impact angle for the given impact probability map
"""
class ImpactProbabilityMap():
    def __init__(self, prob_map, offset, offset_idx, impact_speed, impact_angle):
        self.prob_map = prob_map
        self.offset = offset
        self.offset_idx = offset_idx
        self.impact_speed = impact_speed
        self.impact_angle = impact_angle

"""
Implementation of a single layer of impact map
Args:
    altitude        # Altitude loss of the impact map w.r.t. the fall origin
    distribution    # 2D Gaussian distribution of the impact probability map (heading and distance)
    v_imp           # Predicted impact speed
    angle_imp       # Predicted impact angle
"""
class ImpactLayer():
    def __init__(self, altitude, distribution, v_imp, angle_imp):
        self.altitude = altitude
        self.distribution = distribution
        self.v_imp = v_imp
        self.angle_imp = angle_imp

    """
    Calculate impact probability map (2D array) for the given impact layer.
    Args:
        n_sigma     # Number of standard deviations to be considered during Gaussian PDF sampling
        ground_map  # Ground map to which grid the probability map should be calculated
    Returns:   
        ImpactProbabilityMap    # An impact probability map
    """
    def generate_probability_map(self, n_sigma, ground_map):
        s0 = np.sqrt(self.distribution.cov[0, 0])
        s1 = np.sqrt(self.distribution.cov[1, 1])

        hdg_lims = [self.distribution.mean[0] - n_sigma * s0, self.distribution.mean[0] + n_sigma * s0]
        dist_lims = [self.distribution.mean[1] - n_sigma * s1, self.distribution.mean[1] + n_sigma * s1]

        # Find the x,y coordinate limits of the map
        xlim = [0., 0.]
        ylim = [0., 0.]

        xlim[0] = np.min([dist_lims[0] * np.cos(hdg_lims[0]), dist_lims[0] * np.cos(hdg_lims[1]), dist_lims[1] * np.cos(hdg_lims[0]), dist_lims[1] * np.cos(hdg_lims[1])])
        ylim[0] = np.min([dist_lims[0] * np.sin(hdg_lims[0]), dist_lims[0] * np.sin(hdg_lims[1]), dist_lims[1] * np.sin(hdg_lims[0]), dist_lims[1] * np.sin(hdg_lims[1])])

        xlim[1] = np.max([dist_lims[0] * np.cos(hdg_lims[0]), dist_lims[0] * np.cos(hdg_lims[1]), dist_lims[1] * np.cos(hdg_lims[0]), dist_lims[1] * np.cos(hdg_lims[1])])
        ylim[1] = np.max([dist_lims[0] * np.sin(hdg_lims[0]), dist_lims[0] * np.sin(hdg_lims[1]), dist_lims[1] * np.sin(hdg_lims[0]), dist_lims[1] * np.sin(hdg_lims[1])])

        idx_min = [int(i) for i in ground_map.ENUtoGrid([ylim[0], xlim[0]])]
        idx_max = [int(i) for i in ground_map.ENUtoGrid([ylim[1], xlim[1]])]

        grid_size = [int(i) for i in [idx_max[0] - idx_min[0] + 1, idx_max[1] - idx_min[1] + 1]]

        offset = np.array([ylim[0], xlim[0]])
        offset_idx = idx_min
        prob_map = np.zeros(grid_size)
        for idx_y in range(grid_size[0]):
            for idx_x in range(grid_size[1]):
                cell_idx = np.array(ground_map.GridToENU(np.array([idx_y, idx_x]) + offset_idx))
                hdg = np.arctan2(cell_idx[0], cell_idx[1])
                if abs(self.distribution.mean[0] - hdg) > np.pi:
                    hdg += 2 * np.pi
                dist = np.sqrt(np.sum(cell_idx ** 2))
                prob_map[idx_y, idx_x] = self.distribution.pdf([hdg, dist])

        prob_map = prob_map / np.sum(prob_map)

        return ImpactProbabilityMap(prob_map, offset, offset_idx, self.v_imp, self.angle_imp)  

"""
Class implementing an aircraft and its parameters
Args:
    S       # Aircraft cross-section area [m2]
    m       # aircraft mass [kg]
    c       # drag coefficient [-]
    c_sigma # std. deviation of drag coefficient [-]
    r       # radius of aircraft circumscribed circle [m]
"""
class Aircraft():
    def __init__(self, S, m, c, c_sigma, r):
        self.S = S
        self.m = m
        self.c = c
        self.c_sigma = c_sigma
        self.r = r

"""
ODE for a ballistic fall. Source: https://en.wikipedia.org/wiki/Projectile_motion#Numerical_solution
ODE variable definied as z = [dist, alt, v_horizont, v_vertical]
"""
def BallisticFallODE(z, t, param):
    """
       ODE for a ballistic fall.

       Parameters:
           z (list): State vector [distance, altitude, horizontal velocity, vertical velocity].
           t (float): Current time (not used in this equation as it's time-independent).
           param (tuple): Parameters (g, mu), where:
               g: Gravitational acceleration (e.g., 9.81 m/s^2).
               mu: Drag coefficient term, calculated as CρS/(2m).

       Returns:
           list: Derivatives [d(dist)/dt, d(alt)/dt, d(v_horizontal)/dt, d(v_vertical)/dt].
       """

    # g = param[0]
    # dz = [0, 0, 0, 0]
    # dz[0] = z[2]
    # dz[1] = z[3]
    # dz[2] = 0
    # dz[3] = -g
    # return dz

    g, mu = param  # Extract gravity and drag coefficient
    dist, alt, v_horizont, v_vertical = z  # State variables

    # Compute the speed (magnitude of velocity vector)
    speed = (v_horizont ** 2 + v_vertical ** 2) ** 0.5

    # Define the derivatives
    dz = [0, 0, 0, 0]
    dz[0] = v_horizont  # Change in horizontal position (velocity)
    dz[1] = v_vertical  # Change in vertical position (velocity)
    dz[2] = -mu * v_horizont * speed  # Horizontal velocity change (drag)
    dz[3] = -g - mu * v_vertical * speed  # Vertical velocity change (gravity + drag)

    return dz


"""
Calculate ballistic trajectory based on the given parameters
Args:
    Cd         # drag coefficient
    S          # aircraft cross-section area
    hdg0       # initial heading
    v0         # initial speed
    sim_time   # maximum simulation time
Returns:
    BallisticFall   # generated ballistic fall
"""
def GenerateBallisticFall(Cd, S, m, hdg0, v0, sim_time):
    # rho = 1.17 # kg / m3
    # g = 9.81
    #
    # t = np.linspace(0, sim_time, int(np.round(sim_time / 0.1)) + 1)
    #
    # z_init = [0, 0, np.random.random() * 10, np.random.random() * 2]
    # z = integrate.odeint(BallisticFallODE, z_init, t, args=([g],))
    #
    # dist_h, dist_v, v_h, v_v = z.T
    #
    # return BallisticFall(dist_h, dist_v, v_h, v_v, hdg0)

    # Air density in kg/m^3
    rho = 1.17
    # Gravitational acceleration in m/s^2
    g = 9.81

    # Time discretization
    # np.linspace(start, stop, num)
    # Assuming a time step size of 0.1 seconds
    t = np.linspace(0, sim_time, int(np.round(sim_time / 0.1)) + 1)

    # Mu equation
    mu = (Cd * rho * S) / (2 * m)

    # Initial state: [horizontal position, vertical position, horizontal velocity, vertical velocity]
    z_init = [0, 0, v0, 0]

    # Solve the ODE
    z = integrate.odeint(BallisticFallODE, z_init, t, args=([g, mu],))

    # Extract results
    # Transpose the result to separate variables
    dist_h, dist_v, v_h, v_v = z.T

    # Return BallisticFall object
    return BallisticFall(dist_h, dist_v, v_h, v_v, hdg0)


"""
Generate the given number of ballistic falls considering the normal distributions for drag coefficient and initial hdg
Args:
    num_falls       # number of ballistic falls to be generated
    sim_time        # maximum simulation time
    v0              # initial speed
    hdg_mu          # mean value of initial heading
    hdg_sigma       # std deviation of initial heading
    aircraft        # aircraft model
Returns:
    BallisticFall[]    # array of generated ballistic falls
"""
def GenerateBallisticFalls(num_falls, sim_time, v0, hdg_mu, hdg_sigma, aircraft):
    cs = np.random.normal(aircraft.c, aircraft.c_sigma, num_falls)
    hdgs = np.random.normal(hdg_mu, hdg_sigma, num_falls)
    falls = [None] * num_falls
    for i in range(num_falls):
        falls[i] = GenerateBallisticFall(cs[i], aircraft.S, aircraft.m, hdgs[i], v0, sim_time)
    
    return falls
    
"""
Generate the given impact layers based on the given parameters for simulating the ballistic falls.
Args:
    num_falls       # number of ballistic falls to be generated
    sim_time        # maximum simulation time
    v0              # initial speed
    hdg             # mean value of initial heading
    hdg_sigma       # std deviation of initial heading
    altitude_slices # altitudes w.r.t. fall origin at which the impact maps are created
    aircraft        # aircraft model
Returns:
    ImpactLayer[]   # Array of calculated impact layers

"""
def GenerateImpactMap(num_falls, sim_time, v0, hdg, hdg_sigma, altitude_slices, aircraft):
    falls = GenerateBallisticFalls(num_falls, sim_time, v0, hdg, hdg_sigma, aircraft)
    end_pos = np.zeros([len(altitude_slices), 2, len(falls)])
    impact_v = np.zeros([len(altitude_slices), len(falls)])
    impact_angle = np.zeros([len(altitude_slices), len(falls)])

    # Interpolate fall parameters at the desired altitudes
    # Reverse the arrays as interpolation expects the x-values in increasing order
    for idx, fall in enumerate(falls):
        dist_samples = np.interp(altitude_slices, fall.alt[::-1], fall.dist[::-1])
        v_h_samples = np.interp(altitude_slices, fall.alt[::-1], fall.v_h[::-1])
        v_v_samples = np.interp(altitude_slices, fall.alt[::-1], fall.v_v[::-1])
        hdg_samples = [fall.init_hdg] * len(dist_samples)

        end_pos[:, :, idx] = np.transpose(np.vstack((hdg_samples, dist_samples)))
        impact_v[:, idx] = np.sqrt(np.power(v_v_samples, 2) + np.power(v_h_samples, 2))
        impact_angle[:, idx] = np.arctan2(v_v_samples, v_h_samples)

    # Determine where the aircraft can be and the mean values of impact angle and energy
    impact_layers = [None] * len(altitude_slices)
    for idx, alt in enumerate(altitude_slices):
        v_imp = np.mean(impact_v[idx, :])
        angle_imp = np.mean(impact_angle[idx, :])
        pos = np.transpose(end_pos[idx, :, :])

        mean = np.mean(pos, axis=0)
        cov = np.cov(pos, rowvar=False)

        distribution = multivariate_normal(mean=mean, cov=cov)
        impact_layers[idx] = ImpactLayer(alt, distribution, v_imp, angle_imp)

    return impact_layers
