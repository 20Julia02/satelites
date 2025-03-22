import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.projections.polar import PolarAxes
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

def plot_skyplot_trajectory(
    satelites_epoch: np.ndarray,
    el_mask: float = 10.0,
    epoch_index: int = 0
) -> None:
    
    fontsize = 18
    plt.rc('grid', color='gray', linewidth=1, linestyle='--')
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('font', size=fontsize)

    gnss_systems = {
        "PG": {"label": "GPS",     "color": "#70d6ff",  "count": 0},
        "PE": {"label": "Galileo", "color": "#ff70a6",   "count": 0},
        "PR": {"label": "Glonass", "color": "#ff9770",    "count": 0},
        "PC": {"label": "Beidou",  "color": "#ffd670", "count": 0},
    }

    fig = plt.figure(figsize=(11, 7))
    plt.subplots_adjust(bottom=0.08, top=0.90, left=0.005, right=0.74)
    ax: PolarAxes = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    num_epochs, num_sats, _ = satelites_epoch.shape

    for sat_index in range(num_sats):
        sat_name = None
        start = epoch_index
        while start > 0:
            entry = satelites_epoch[start - 1, sat_index]
            if entry[0] is None or np.isnan(entry[4]) or entry[4] <= el_mask:
                break
            start -= 1
        end = epoch_index
        while end < num_epochs - 1:
            entry = satelites_epoch[end + 1, sat_index]
            if entry[0] is None or np.isnan(entry[4]) or entry[4] <= el_mask:
                break
            end+=1

        trajectory = []
        for t in range(start, end + 1):
            entry = satelites_epoch[t, sat_index]
            if entry[0] is not None and not np.isnan(entry[4]) and not np.isnan(entry[5]) and entry[4] >= el_mask:
                sat_name = entry[0]
                az = np.radians(entry[5])
                r = 90 - entry[4]
                trajectory.append((az, r))
        if not trajectory or sat_name is None:
            continue

        system_prefix = sat_name[:2]
        system = gnss_systems.get(system_prefix)

        trajectory = np.array(trajectory)

        current = satelites_epoch[epoch_index, sat_index]
        if system:
            if current[0] is not None and not np.isnan(current[4]) and not np.isnan(current[5]) and current[4] > el_mask:
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=system["color"], linewidth=1.0, alpha=0.6)
                az = np.radians(current[5])
                r = 90 - current[4]
                ax.plot(az, r, marker='o', color=system["color"], markersize=5)
                ax.annotate(current[0][1:], xy=(az, r), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.05", fc=system["color"], alpha=0.7), fontsize=10)
                system["count"] += 1

    mask_radius = 90 - el_mask
    theta = np.linspace(0, 2*np.pi, 360)
    ax.plot(theta, [mask_radius]*len(theta), color = "purple", linestyle='-.', linewidth=2)

    all_count = sum(sys["count"] for sys in gnss_systems.values())
    legend_handles = [mpatches.Patch(color='white', label=f"{all_count:02d}  ALL")]

    for system in gnss_systems.values():
        legend_handles.append(
            mpatches.Patch(color=system["color"], label=f"{system['count']:02d}  {system['label']}")
        )
    legend_handles.append(Line2D([0],[0], color="purple", linestyle='-.', linewidth=2, label=f"Maska {el_mask}°"))
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.1, 1.05), loc='upper left', borderaxespad=0.0)

    ax.set_yticks(range(0, 91, 30))
    ax.set_yticklabels(['', '', '', ''])

    plt.show()

def plot_dop(dop_dict: dict):
    time = dop_dict.keys()
    gdop_list = [dop_dict[t][0] for t in time]
    pdop_list = [dop_dict[t][1] for t in time]
    tdop_list = [dop_dict[t][2] for t in time]
    hdop_list = [dop_dict[t][3] for t in time]
    vdop_list = [dop_dict[t][4] for t in time]

    plt.figure(figsize=(14, 6))

    plt.plot(time, gdop_list, label='GDOP', linewidth=2)
    plt.plot(time, pdop_list, label='PDOP', linewidth=2)
    plt.plot(time, tdop_list, label='TDOP', linewidth=2)
    plt.plot(time, hdop_list, label='HDOP', linewidth=2)
    plt.plot(time, vdop_list, label='GVDOP', linewidth=2)

    def format_minutes(x, _):
        h = int(x) // 60
        m = int(x) % 60
        return f"{h:02d}:{m:02d}"
    
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_minutes))
    plt.xticks(np.arange(0, 24 * 60 + 1, 60*3))
    plt.xlim(0, 24 * 60)
    plt.ylim(bottom=0)
    plt.xlabel('Czas [godziny]')
    plt.ylabel('Wartość DOP')
    plt.grid(True)
    plt.legend()
    # plt.tight_layout()
    plt.show()