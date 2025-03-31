import sys
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QTabWidget, QLineEdit,
    QLabel, QPushButton, QSpinBox, QFormLayout, QHBoxLayout, QGroupBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from charts import plot_visibility_radius, plot_positions_map, plot_skyplot_trajectory, plot_dop, plot_num_sats, plot_visibility, plot_elevations
from sat_calc import Satelite, Observer, calc_dop, sat_visibility_intervals
from transformations import ymd_to_gps
from alm_module import get_alm_data
import numpy as np

class ParamsTab(QWidget):
    apply_clicked = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        main_layout = QFormLayout()

        location_group = QGroupBox("Pozycja obserwatora")
        location_layout = QFormLayout()
        self.lat_input = QLineEdit()
        self.lon_input = QLineEdit()
        self.h_input = QLineEdit()
        self.lat_input.setText("50")
        self.lon_input.setText("20")
        self.h_input.setText("100")

        location_layout.addRow("Szerokość geograficzna [°]:", self.lat_input)
        location_layout.addRow("Długość geograficzna [°]:", self.lon_input)
        location_layout.addRow("Wysokość n.p.m. [m]:", self.h_input)
        location_group.setLayout(location_layout)

        parameter_group = QGroupBox("Czas i maska elewacji")
        parameter_layout = QFormLayout()
        self.elev_cutoff_input = QSpinBox()
        self.elev_cutoff_input.setRange(0, 89)
        self.day = QtWidgets.QDateEdit()
        self.day.setDisplayFormat("yyyy-MM-dd")
        self.day.setCalendarPopup(True)
        self.day.setDate(QtCore.QDate.currentDate())
        self.day.setMinimumDate(QtCore.QDate(2025, 1, 1))
        self.day.setMaximumDate(QtCore.QDate(2025, 12, 1))

        parameter_layout.addRow("Maska elewacji [°]:", self.elev_cutoff_input)
        parameter_layout.addRow("Wybierz datę: ", self.day)
        parameter_group.setLayout(parameter_layout)

        input_group = QGroupBox("Dane wejściowe")
        input_layout = QFormLayout()
        self.almanac_path_input = QLineEdit()
        self.browse_btn = QPushButton("...")
        self.browse_btn.setFixedWidth(30)
        self.browse_btn.clicked.connect(self.browse_file)

        file_selector_layout = QHBoxLayout()
        file_selector_layout.addWidget(self.almanac_path_input)
        file_selector_layout.addWidget(self.browse_btn)

        input_layout.addRow("Plik almanacha (.alm):", file_selector_layout)
        input_group.setLayout(input_layout)

        systems_group = QGroupBox("Systemy GNSS")
        systems_layout = QHBoxLayout()
        self.gps_check = QtWidgets.QCheckBox("GPS")
        self.gps_check.setChecked(True)
        self.glonass_check = QtWidgets.QCheckBox("GLONASS")
        self.glonass_check.setChecked(True)
        self.galileo_check = QtWidgets.QCheckBox("GALILEO")
        self.galileo_check.setChecked(True)
        self.beidou_check = QtWidgets.QCheckBox("BEIDOU")
        self.beidou_check.setChecked(True)

        systems_layout.addWidget(self.gps_check)
        systems_layout.addWidget(self.glonass_check)
        systems_layout.addWidget(self.galileo_check)
        systems_layout.addWidget(self.beidou_check)
        systems_group.setLayout(systems_layout)

        self.apply_button = QPushButton(text= "Apply", parent=self)
        self.apply_button.setFixedSize(100,30)
        self.apply_button.clicked.connect(self.handle_apply)

        main_layout.addWidget(location_group)
        main_layout.addWidget(parameter_group)
        main_layout.addWidget(input_group)
        main_layout.addWidget(systems_group)
        main_layout.addWidget(self.apply_button)

        self.setLayout(main_layout)

    def browse_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Wybierz plik almanacha",
            "",
            "Pliki almanacha (*.alm);;Wszystkie pliki (*)"
        )
        if file_path:
            self.almanac_path_input.setText(file_path)
    
    def validate_inputs(self):
        valid = True
        inputs = [self.lat_input, self.lon_input, self.h_input, self.almanac_path_input]

        for input in inputs:
            if not input.text().strip():
                input.setStyleSheet("background-color: #ffcccc;")
                valid = False
            else:
                input.setStyleSheet("")

        return valid

    
    def handle_apply(self):
        if not self.validate_inputs():
            return
        systems = set()
        if self.gps_check.isChecked():
            systems.add(0)
        if self.glonass_check.isChecked():
            systems.add(1)
        if self.galileo_check.isChecked():
            systems.add(2)
        if self.beidou_check.isChecked():
            systems.add(3)
        
        qdate = self.day.date()
        data = {
            "lat": float(self.lat_input.text()),
            "lon": float(self.lon_input.text()),
            "h": float(self.h_input.text()),
            "el_mask": self.elev_cutoff_input.value(),
            "date": (qdate.year(), qdate.month(), qdate.day()),
            "almanac_path": self.almanac_path_input.text(),
            "systems": systems
        }
        self.apply_clicked.emit(data)

class SkyPlotTab(QWidget):
    def __init__(self):
        super().__init__()

        self.satelites_epoch = None
        self.el_mask = 0

        self.figure = Figure(figsize=(14, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(6*24-1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.valueChanged.connect(self.slider_moved)

        self.time_label = QLabel("Godzina: 00:00")
        self.time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.time_label)
        layout.addWidget(self.slider)

        self.setLayout(layout)

    def update_data(self, satelites_epoch, el_mask):
        self.satelites_epoch = satelites_epoch
        self.el_mask = el_mask
        self.slider.setValue(0)
        self.draw_skyplot(0)

    def slider_moved(self, value):
        self.draw_skyplot(value)

    def draw_skyplot(self, minute_index):
        if self.satelites_epoch is None:
            return

        hour = minute_index * 10 // 60
        minute = (minute_index * 10) % 60
        time_str = f"{hour:02d}:{minute:02d}"
        self.time_label.setText(f"Godzina: {time_str}")

        plot_skyplot_trajectory(self.figure, self.satelites_epoch, self.el_mask, minute_index * 10)
        self.canvas.draw()


class ChartsTab(QWidget):
    def __init__(self):
        super().__init__()

        self.satelites_epoch = None
        self.el_mask = 0
        self.dop_dict = None
        self.visible_sats_per_minute = None
        self.sat_type = None

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)

        self.inner_widget = QWidget()
        self.inner_layout = QVBoxLayout(self.inner_widget)
        self.inner_layout.setSpacing(30)

        scroll_area.setWidget(self.inner_widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

    def update_data(self, satelites_epoch, el_mask, dop_dict, visible_sats_per_minute, sat_type):
        self.satelites_epoch = satelites_epoch
        self.el_mask = el_mask
        self.dop_dict = dop_dict
        self.visible_sats_per_minute = visible_sats_per_minute
        self.sat_type = sat_type
        self.draw_charts()

    def draw_charts(self):
        while self.inner_layout.count():
            item = self.inner_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        if self.dop_dict:
            fig_dop =  Figure(figsize=(9, 5), constrained_layout=True)
            canvas_dop = FigureCanvasQTAgg(fig_dop)
            plot_dop(fig_dop, self.dop_dict)
            self.inner_layout.addWidget(canvas_dop)
        if self.visible_sats_per_minute and self.sat_type:
            fig_num_sats = Figure(figsize=(9, 5), constrained_layout=True)
            canvas_num_sats = FigureCanvasQTAgg(fig_num_sats)
            plot_num_sats(fig_num_sats, self.visible_sats_per_minute, self.sat_type)
            self.inner_layout.addWidget(canvas_num_sats)
        if self.satelites_epoch is None:
            return
        fig_elev = Figure(figsize=(9, 8), constrained_layout=True)
        canvas_elev = FigureCanvasQTAgg(fig_elev)
        plot_elevations(fig_elev, self.satelites_epoch, self.el_mask)
        self.inner_layout.addWidget(canvas_elev)

        sat_vis = sat_visibility_intervals(self.satelites_epoch, self.el_mask)
        fig_visibility = Figure(figsize=(9, len(sat_vis) * 0.06), constrained_layout=True)
        canvas_visibility = FigureCanvasQTAgg(fig_visibility)
        plot_visibility(fig_visibility, sat_vis)
        self.inner_layout.addWidget(canvas_visibility)

        total_height = self.inner_layout.sizeHint().height()
        self.inner_widget.setMinimumHeight(total_height)


class MapTab(QWidget):
    def __init__(self):
        super().__init__()

        self.satelites_epoch = None
        self.observer = None
        self.el_mask = 0

        self.figure = Figure(figsize=(14, 8), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(6*24-1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.valueChanged.connect(self.slider_moved)

        self.time_label = QLabel("Godzina: 00:00")
        self.time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.time_label)
        layout.addWidget(self.slider)

        self.setLayout(layout)
    
    def update_data(self, satelites_epoch, el_mask, observer):
        self.satelites_epoch = satelites_epoch
        self.el_mask = el_mask
        self.observer = observer
        self.slider.setValue(0)
        self.draw_position_map(0)
    
    def slider_moved(self, value):
         self.draw_position_map(value)
        
    def draw_position_map(self, minute_index):
        if self.satelites_epoch is None or self.observer is None:
            return
        
        hour = minute_index * 10 // 60
        minute = (minute_index * 10) % 60
        time_str = f"{hour:02d}:{minute:02d}"
        self.time_label.setText(f"Godzina: {time_str}")

        plot_positions_map(self.figure, self.satelites_epoch, self.observer, minute_index, self.el_mask)
        self.canvas.draw()


class MapRadiusTab(QWidget):
    def __init__(self):
        super().__init__()

        self.satelites_epoch = None
        self.el_mask = 0

        self.figure = Figure(figsize=(14, 8), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(6*24-1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.valueChanged.connect(self.slider_moved)

        self.time_label = QLabel("Godzina: 00:00")
        self.time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.time_label)
        layout.addWidget(self.slider)

        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.redraw_map)

        self.pending_value = None

        self.setLayout(layout)
    
    def update_data(self, satelites_epoch, el_mask):
        self.satelites_epoch = satelites_epoch
        self.el_mask = el_mask
        self.slider.setValue(0)
        self.draw_sats_map(0)
    
    def slider_moved(self, value):
        self.pending_value = value
        self.timer.start(100)
    
    def redraw_map(self):
        if self.pending_value is not None:
            self.draw_sats_map(self.pending_value)
            self.pending_value = None
        
    def draw_sats_map(self, minute_index):
        if self.satelites_epoch is None:
            return
        hour = minute_index * 10 // 60
        minute = (minute_index * 10) % 60
        time_str = f"{hour:02d}:{minute:02d}"
        self.time_label.setText(f"Godzina: {time_str}")
        try:
            plot_visibility_radius(self.figure, self.satelites_epoch, minute_index, self.el_mask)
            self.canvas.draw()
        except Exception as e:
            print(f"[BŁĄD RYSOWANIA MAPY WIDOCZNOŚCI]: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GNNS Calculator")
        self.setMinimumSize(1100, 800)

        tabs = QTabWidget()
        self.params_tab = ParamsTab()
        self.charts_tab = ChartsTab()
        self.plot_tab = SkyPlotTab()
        self.map_tab = MapTab()
        self.map_radius_tab = MapRadiusTab()
        
        tabs.addTab(self.params_tab, "Parametry")
        tabs.addTab(self.plot_tab, "Sky Plot")
        tabs.addTab(self.charts_tab, "Wykresy")
        tabs.addTab(self.map_tab, "Ground Track")
        tabs.addTab(self.map_radius_tab, "Mapa widoczności satelitów")

        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        
        central_widget.setLayout(layout)
        
        self.setCentralWidget(central_widget)

        self.params_tab.apply_clicked.connect(self.process_inputs)

    def process_inputs(self, input: dict):
        satelites_epoch, dop_dict, visible_sats_per_minute = self.calculate_satelites(input)
        self.plot_tab.update_data(satelites_epoch, input["el_mask"])
        self.charts_tab.update_data(satelites_epoch, input["el_mask"], dop_dict, visible_sats_per_minute, input["systems"])
        self.map_tab.update_data(satelites_epoch, input["el_mask"], (input["lat"], input["lon"]))
        self.map_radius_tab.update_data(satelites_epoch, input["el_mask"])
    
    def calculate_satelites(self, input: dict):
        sat_type = input["systems"]
        nav_data = get_alm_data(input["almanac_path"])
        observer = Observer(input["lat"], input["lon"], input["h"])
        el_mask = input["el_mask"]
        year, month, day = input["date"]

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
            _, sec_week = ymd_to_gps([year, month, day, hour, minute, second])
            for iter2 in range(len(satelites)):
                x_ecef, y_ecef, z_ecef = satelites[iter2].calculate_position(sec_week)
                azimuth, elevation, r, dXYZ = Satelite.calc_topo_cord(np.array([x_ecef, y_ecef, z_ecef]), (observer.phi, observer.lam, observer.h))
                satelites_epoch[int(time_step / (10 * 60)), iter2] = [satelites[iter2].sat_name, x_ecef, y_ecef, z_ecef, elevation, azimuth]
                # DOP
                if elevation > el_mask:
                    prefix = satelites[iter2].sat_name[:2]
                    if prefix in visible_count:
                        visible_count[prefix] += 1
                    A_row = [-dXYZ[0] / r,
                            -dXYZ[1] / r,
                            -dXYZ[2] / r,
                            1]
                    A_list.append(A_row)
            if len(A_list) >= 4:
                A_matrix = np.array(A_list)
                dop_dict[time_step] = calc_dop(A_matrix, observer.R_neu)
            visible_sats_per_minute[time_step] = visible_count
        return satelites_epoch, dop_dict, visible_sats_per_minute

 
if __name__ =="__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())