from matplotlib.figure import Figure
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.projections.polar import PolarAxes
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from pyproj import Transformer
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_skyplot_trajectory(
    fig: Figure,
    satelites_epoch: np.ndarray,
    el_mask: float = 10.0,
    minute: int = 0
) -> None:
    """
    creates skyplot trajectory chart.
    Input should be array 3D [epoch x satelites x [x, y, z, elevation, azimuth]]
    """
    fontsize = 16
    if fig.axes:
        fig.delaxes(fig.axes[0])

    plt.rc('grid', color='gray', linewidth=1, linestyle='--')
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('font', size=fontsize)

    gnss_systems = {
        "PG": {"label": "GPS",     "color": "#70d6ff",  "count": 0},
        "PE": {"label": "Galileo", "color": "#ff70a6",  "count": 0},
        "PR": {"label": "Glonass", "color": "#ff9770",  "count": 0},
        "PC": {"label": "Beidou",  "color": "#c77dff",  "count": 0},
    }

    ax: PolarAxes = fig.add_subplot(111, polar=True)
    ax.set_position([0.1, 0.1, 0.65, 0.75])
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    num_epochs, num_sats, _ = satelites_epoch.shape

    for sat_index in range(num_sats):
        sat_name = None
        start = minute//10
        while start > 0:
            entry = satelites_epoch[start - 1, sat_index]
            if entry[0] is None or np.isnan(entry[4]) or entry[4] <= el_mask:
                break
            start -= 1
        end = minute//10
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

        current = satelites_epoch[minute//10, sat_index]
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
    ax.plot(theta, [mask_radius]*len(theta), color = "#e63946", linestyle='-.', linewidth=2)

    all_count = sum(sys["count"] for sys in gnss_systems.values())
    legend_handles = [mpatches.Patch(color='white', label=f"{all_count:02d}  ALL")]

    for system in gnss_systems.values():
        legend_handles.append(
            mpatches.Patch(color=system["color"], label=f"{system['count']:02d}  {system['label']}")
        )
    legend_handles.append(Line2D([0],[0], color="#e63946", linestyle='-.', linewidth=2, label=f"Maska {el_mask}°"))
    ax.legend(handles=legend_handles, 
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=5,
            fontsize=9)

    ax.set_yticks(range(0, 91, 30))
    ax.set_yticklabels(['', '', '', ''])

def plot_dop(fig: Figure, dop_dict: dict):

    fig.clear()
    ax = fig.add_subplot(111)

    time = sorted([t / 60 for t in dop_dict.keys()])
    gdop_list = [dop_dict[t*60][0] for t in time]
    pdop_list = [dop_dict[t*60][1] for t in time]
    tdop_list = [dop_dict[t*60][2] for t in time]
    hdop_list = [dop_dict[t*60][3] for t in time]
    vdop_list = [dop_dict[t*60][4] for t in time]

    ax.plot(time, gdop_list, label='GDOP', linewidth=2, color = "#70d6ff")
    ax.plot(time, pdop_list, label='PDOP', linewidth=2, color = "#ff70a6")
    ax.plot(time, tdop_list, label='TDOP', linewidth=2, color = "#ff9770")
    ax.plot(time, hdop_list, label='HDOP', linewidth=2, color = "#c77dff")
    ax.plot(time, vdop_list, label='VDOP', linewidth=2, color = "#1982c4")

    def format_minutes(x, _):
        h = int(x) // 60
        m = int(x) % 60
        return f"{h:02d}:{m:02d}"
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_minutes))
    ax.set_xticks(np.arange(0, 24 * 60 + 1, 60*3))
    ax.set_xlim(0, 24 * 60)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Czas [godziny]')
    ax.set_ylabel('Wartość DOP')
    ax.grid(True)
    ax.legend()
    ax.set_title('Wykres wartości DOP w czasie')
    ax.grid(True)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        fontsize=9
    )
    fig.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.12)


def plot_num_sats(fig: Figure, time_sats_dict: dict, sat_list):
    
    fig.clear()
    ax = fig.add_subplot(111)

    system_map = {
        0: ('PG', 'GPS',    "#70d6ff"),
        1: ('PR', 'GLONASS',"#ff70a6"),
        2: ('PE', 'GALILEO',"#ff9770"),
        3: ('PC', 'BEIDOU' ,"#c77dff")
    }

    time = sorted([t / 60 for t in time_sats_dict.keys()])
    
    sum_sats = np.zeros(len(time))

    for sat_type in sat_list:
        if sat_type in system_map:
            prefix, label, color = system_map[sat_type]
            values = [time_sats_dict[int(t * 60)][prefix] for t in time]
            ax.plot(time, values, label=label, linewidth=2, color = color)
            sum_sats += np.array(values)
    
    ax.plot(time, sum_sats, label='Wszystkie', linewidth=2.5, color='#1982c4')

    def format_minutes(x, _):
        h = int(x) // 60
        m = int(x) % 60
        return f"{h:02d}:{m:02d}"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_minutes))
    ax.set_xticks(np.arange(0, 24 * 60 + 1, 180))
    ax.set_xlim(0, 24 * 60)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Czas [godziny]')
    ax.set_ylabel('Liczba satelitów')
    ax.grid(True)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        fontsize=9
    )
    ax.set_title('Wykres liczby satelitów w czasie')
    fig.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.12)


def plot_visibility(fig: Figure, sat_visibility: dict):
    fig.clear()
    gnss_systems = {
        "PG": {"label": "GPS",     "color": "#70d6ff"},
        "PE": {"label": "Galileo", "color": "#ff70a6"},
        "PR": {"label": "Glonass", "color": "#ff9770"},
        "PC": {"label": "Beidou",  "color": "#c77dff"},
    }
    ax = fig.add_subplot(111)
    yticks = []
    ylabels = []

    for i, (sat_name, intervals) in enumerate(sorted(sat_visibility.items(), reverse=True)):
        bars = [(start, end - start) for start, end in intervals]
        system_prefix = sat_name[:2]
        color = gnss_systems[system_prefix]["color"]
        ax.broken_barh(bars, (i-0.3, 0.6), facecolors=color, zorder=2)
        yticks.append(i)
        if i % 5 == 0:
            ylabels.append(sat_name)
        else:
            ylabels.append('')
    ax.set_ylim(-0.4, len(sat_visibility) - 0.4)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("Czas [godziny]")
    ax.set_xlim(0, 1440)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(180))

    def format_minutes(x, _):
        h = int(x) // 60
        m = int(x) % 60
        return f"{h:02d}:{m:02d}"
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_minutes))
    ax.grid(True, zorder=1)
    ax.set_title("Okna widoczności satelitów")
    fig.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.12)

def plot_elevations(fig: Figure, satelites_epoch, el_mask=10):

    fig.clear()
    ax = fig.add_subplot(111)

    ax.axhline(y=el_mask, color='#e63946', linestyle='--', linewidth=1.5, label=f'Maska {el_mask}°')

    num_epochs, num_sats, _ = satelites_epoch.shape
    time = [t * 10 for t in range(num_epochs)] 
    
    for sat_index in range(satelites_epoch.shape[1]):
        sat_name = None
        sat_elev = []

        for epoch_index in range(num_epochs):
            row = satelites_epoch[epoch_index, sat_index]
            if row[0] is not None and not np.isnan(row[4]):
                elevation = row[4]
                sat_name = row[0]
                sat_elev.append(elevation if elevation > el_mask else np.nan)
            else:
                sat_elev.append(np.nan)
        if sat_name:
            ax.plot(time, sat_elev, label = sat_name, linewidth=1)

    def format_minutes(x, _):
        h = int(x) // 60
        m = int(x) % 60
        return f"{h:02}:{m:02}"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_minutes))
    ax.set_xticks(np.arange(0, 1441, 180))
    ax.set_xlim(0, 1440)
    ax.set_ylim(0, 90)
    ax.set_xlabel('Czas [godziny]')
    ax.set_ylabel('Elewacja [°]')
    ax.set_title('Elewacja satelitów w czasie')
    ax.grid(True)
    ax.legend(
    ncol=12,
    fontsize=7,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15)
    )
    fig.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.12)


def plot_positions_map(fig, satelites_epoch, observer, minute=0, el_mask=0):
    
    fig.clear()

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(color='darkgray', linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, color='#ffffff6a', linestyle='-', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    ax.add_feature(cfeature.BORDERS, linestyle=':', color="lightgrey")
    ax.add_feature(cfeature.LAND, facecolor='#f0ede6')
    ax.add_feature(cfeature.OCEAN, facecolor='#94c5eb')

    ax.plot(observer[1], observer[0], marker='v', color='#d90429',
        markersize=10, transform=ccrs.Geodetic(), label="Obserwator")

    ax.set_title("Pozycje satelitów na mapie", fontsize=14)

    gnss_systems = {
        "PG": {"label": "GPS",     "color": "#70d6ff"},
        "PE": {"label": "Galileo", "color": "#ff70a6"},
        "PR": {"label": "Glonass", "color": "#ff9770"},
        "PC": {"label": "Beidou",  "color": "#c77dff"},
    }

    ecef2geodetic = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
    num_epochs, num_sats, _ = satelites_epoch.shape

    for sat_index in range(num_sats):
        sat_name = None
        start = minute
        while start > 0:
            entry = satelites_epoch[start - 1, sat_index]
            if entry[0] is None or np.isnan(entry[4]) or entry[4] <= el_mask:
                break
            start -= 1
        end = minute
        while end < num_epochs - 1:
            entry = satelites_epoch[end + 1, sat_index]
            if entry[0] is None or np.isnan(entry[4]) or entry[4] <= el_mask:
                break
            end+=1

        trajectory = []
        for t in range(start, end + 1):
            entry = satelites_epoch[t, sat_index]
            if entry[0] is not None and not np.isnan(entry[4]) and entry[4] >= el_mask:
                sat_name = entry[0]
                lon, lat, alt = ecef2geodetic.transform(entry[1], entry[2], entry[3])
                trajectory.append((lon, lat))
        if not trajectory or sat_name is None:
            continue

        system_prefix = sat_name[:2]
        system = gnss_systems.get(system_prefix)
        trajectory = np.array(trajectory)

        if len(trajectory) > 1 and system:
            lons, lats = zip(*trajectory)
            ax.plot(lons, lats, color=system["color"], linewidth=1, alpha=0.6, transform=ccrs.Geodetic())

        current = satelites_epoch[minute, sat_index]
        if current[0] is not None and system:
            if current[4] > el_mask:
                lon, lat, alt = ecef2geodetic.transform(current[1], current[2], current[3])
                ax.plot(lon, lat, 'o', color=system['color'], markersize=5, transform=ccrs.Geodetic())
                ax.annotate(current[0][1:], xy=(lon, lat), transform=ccrs.Geodetic(), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.05", fc=system["color"], alpha=0.7), fontsize=10)
