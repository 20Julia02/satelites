from transformations import ymd_to_gps
from alm_module import get_alm_data
import numpy as np
from charts import plot_skyplot_trajectory, plot_dop, plot_num_sats, plot_visibility, plot_elevations
from sat_calc import Satelite, Observer, calc_dop, sat_visibility_intervals


nav_data = get_alm_data('almanac.alm')

sat_type = {0, 1, 2, 3}
observer = Observer(52, 21, 100)
el_mask = 10
date = [2025, 2, 27]

satelites = [Satelite(row) for row in nav_data if int(row[0] / 100) in sat_type]
satelites_epoch = np.empty((144, nav_data.shape[0], 6), dtype=object)
dop_dict = {}
visible_sats_per_minute = {}

for time_step in range(0, 24*60*60, 10*60):
    A_list = []
    visible_count = {"PG": 0, "PR": 0, "PC":0, "PE":0}
    hour = time_step // 3600
    minute = (time_step % 3600) // 60
    second = time_step % 60
    _, sec_week = ymd_to_gps([date[0], date[1], date[2], hour, minute, second])
    for iter2 in range(len(satelites)):
        x_ecef, y_ecef, z_ecef = satelites[iter2].calculate_position(sec_week)
        azimuth, elevation, r, dXYZ = Satelite.calc_topo_cord(np.array([x_ecef, y_ecef, z_ecef]), (observer.phi, observer.lam, observer.h))
        satelites_epoch[int(time_step / (10 * 60)), iter2] = [satelites[iter2].sat_name, x_ecef, y_ecef, z_ecef, elevation, azimuth]
        # DOP
        if elevation > el_mask:
            visible_count[satelites[iter2].sat_name[:2]] += 1
            A_row = [-dXYZ[0] / r,
                     -dXYZ[1] / r,
                     -dXYZ[2] / r,
                     1]
            A_list.append(A_row)
    if len(A_list) >= 4:
        A_matrix = np.array(A_list)
        dop_dict[time_step] = calc_dop(A_matrix, observer.R_neu)
    visible_sats_per_minute[time_step] = visible_count

# plot_dop(dop_dict)
# plot_skyplot_trajectory(sxatelites_epoch, el_mask, 0)
# plot_num_sats(visible_sats_per_minute, sat_type)
# sat_vis = sat_visibility_intervals(satelites_epoch, el_mask)
# plot_visibility(sat_vis)
# plot_elevations(satelites_epoch, el_mask)