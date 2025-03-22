from transformations import geodetic_to_cartesian, rotation_matrix_neu
import numpy as np
import math

class Satelite:
    MI = 3.986005 * 10**14
    OMEGA_E = 7.2921151467 * 10**(-5)

    def __init__(self, sat_data):
        self.satelite_id = sat_data[0]
        self.sat_name = self.satelite_type()
        self.e_eccentr = sat_data[2]
        self.a_semi_major_axis = sat_data[3]**2
        self.omega_0_asc_node = math.radians(sat_data[4])
        self.omega_arg_perigee = math.radians(sat_data[5])
        self.m0_mean_anomaly = math.radians(sat_data[6])
        self.time_of_almanac = sat_data[7]
        self.i_orb_inclin = math.radians(54 + sat_data[8])
        self.omega_dot_right_ascen = math.radians(sat_data[9]/1000)
    
    def satelite_type(self):
        if self.satelite_id//100 == 0:
            sat_type = "PG"
        elif self.satelite_id//100 == 1:
            sat_type = "PR"
        elif self.satelite_id//100 == 2:
            sat_type = "PE"
        elif self.satelite_id//100 == 3:
            sat_type = "PC"
        num = int(self.satelite_id%100)
        return f"{sat_type}{num:02}"

    def calculate_position(self, sec_week):
        n_mean_motion = math.sqrt(Satelite.MI/self.a_semi_major_axis**3) 
        tk_elapsed = sec_week - self.time_of_almanac

        mk_anomally_epoch = self.m0_mean_anomaly + n_mean_motion * tk_elapsed
        eccentr_anomaly = self.solve_eccentric_anomaly(mk_anomally_epoch)

        vk_true_anomaly = math.atan2(math.sqrt(1-self.e_eccentr**2)*math.sin(eccentr_anomaly), math.cos(eccentr_anomaly) - self.e_eccentr)

        phi_latitude_argument = vk_true_anomaly + self.omega_arg_perigee
        
        rk_orbit_radius = self.a_semi_major_axis * (1 - self.e_eccentr * math.cos(eccentr_anomaly))

        x_orbite = rk_orbit_radius * math.cos(phi_latitude_argument)
        y_orbite = rk_orbit_radius * math.sin(phi_latitude_argument)

        omega_0_asc_node_corr = self.omega_0_asc_node + (self.omega_dot_right_ascen - Satelite.OMEGA_E) * tk_elapsed - Satelite.OMEGA_E * self.time_of_almanac

        return self.ecef_position((x_orbite, y_orbite), omega_0_asc_node_corr)

    def solve_eccentric_anomaly(self, mk_anomally_epoch, tolerance=1e-9):
        eccentr_anomaly = mk_anomally_epoch
        while True:
             eccentr_anomaly_next = mk_anomally_epoch + self.e_eccentr * math.sin(eccentr_anomaly)
             if abs(eccentr_anomaly_next - eccentr_anomaly) < tolerance:
                 break
             eccentr_anomaly = eccentr_anomaly_next
        return eccentr_anomaly
    
    def ecef_position(self, orb_coords, omega_0_asc_node_corr):
        x_orbite, y_orbite = orb_coords
        x_ecef = x_orbite * math.cos(omega_0_asc_node_corr) - y_orbite*math.cos(self.i_orb_inclin)*math.sin(omega_0_asc_node_corr)
        y_ecef = x_orbite * math.sin(omega_0_asc_node_corr) + y_orbite*math.cos(self.i_orb_inclin)*math.cos(omega_0_asc_node_corr)
        z_ecef = y_orbite*math.sin(self.i_orb_inclin)
        return x_ecef, y_ecef, z_ecef
        
    @staticmethod
    def calc_topo_cord(xyz_ecef, plh_observer):
        xyz = geodetic_to_cartesian(plh_observer[0], plh_observer[1],  plh_observer[2])
        xyz_observer = np.array(xyz)
        R_neu = rotation_matrix_neu(plh_observer[0], plh_observer[1])

        dXYZ = xyz_ecef - xyz_observer
        r = np.linalg.norm(dXYZ)
        n,e, u = R_neu.T @ dXYZ

        distance = np.sqrt(n**2 + e**2 + u**2)
        azimuth = (np.arctan2(e, n))*180/np.pi
        azimuth = azimuth if azimuth > 0 else azimuth + 360
        elevation = np.arcsin(u/distance)*180/np.pi
        return azimuth, elevation, r, dXYZ
    
def calc_dop(array_A, R_neu):
    array_Q = np.linalg.inv(array_A.T@array_A)
    gdop = np.sqrt(np.trace(array_Q))
    pdop = np.sqrt(array_Q[0][0] + array_Q[1][1] + array_Q[2][2])
    tdop = np.sqrt(array_Q[3][3])
    Qneu = R_neu.T@array_Q[0:3, 0:3]@R_neu
    hdop = np.sqrt(Qneu[0][0] + Qneu[1][1])
    vdop = np.sqrt(Qneu[2][2])
    return gdop, pdop, tdop, hdop, vdop

class Observer:
    def __init__(self, phi, lam, h):
        self.phi = math.radians(phi) 
        self.lam = math.radians(lam) 
        self.h = h
        self.xyz = np.array(geodetic_to_cartesian(self.phi, self.lam, self.h))
        self.R_neu = rotation_matrix_neu(self.phi, self.lam)