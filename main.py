import sys
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QTabWidget, QLineEdit,
    QLabel, QPushButton, QSpinBox, QFormLayout, QHBoxLayout, QGroupBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from charts import plot_skyplot_trajectory, plot_dop, plot_num_sats, plot_visibility, plot_elevations
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
        self.time = QtWidgets.QTimeEdit()

        parameter_layout.addRow("Maska elewacji [°]:", self.elev_cutoff_input)
        parameter_layout.addRow("Wybierz datę: ", self.day)
        parameter_layout.addRow("Wybierz godzinę: ", self.time)
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
        self.glonass_check = QtWidgets.QCheckBox("GLONASS")
        self.galileo_check = QtWidgets.QCheckBox("GALILEO")
        self.beidou_check = QtWidgets.QCheckBox("BEIDOU")

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
    
    def handle_apply(self):
        systems = set()
        if self.gps_check.isChecked():
            systems.add(0)
        if self.glonass_check.isChecked():
            systems.add(1)
        if self.galileo_check.isChecked():
            systems.add(2)
        if self.beidou_check.isChecked():
            systems.add(3)
        data = {
            "lat": float(self.lat_input.text()),
            "lon": float(self.lon_input.text()),
            "h": float(self.h_input.text()),
            "el_mask": self.elev_cutoff_input.value(),
            "date": self.day.date().getDate(),
            "time": self.time.time().toString("HH:mm"),
            "almanac_path": self.almanac_path_input.text(),
            "systems": systems
        }
        self.apply_clicked.emit(data)

class SkyPlotTab(QWidget):
    def __init__(self):
        super().__init__()

        self.satelites_epoch = None
        self.el_mask = 0

        self.figure = Figure()
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GNNS Calculator")
        self.setMinimumSize(600, 400)

        tabs = QTabWidget()
        self.params_tab = ParamsTab()
        self.plot_tab = SkyPlotTab()
        
        tabs.addTab(self.params_tab, "Parametry")
        tabs.addTab(self.plot_tab, "Sky Plot")

        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        
        central_widget.setLayout(layout)
        
        self.setCentralWidget(central_widget)

        self.params_tab.apply_clicked.connect(self.process_inputs)

    def process_inputs(self, input: dict):
        satelites_epoch, dop_dict, visible_sats_per_minute = self.calculate_satelites(input)
        sat_vis = sat_visibility_intervals(satelites_epoch, input["el_mask"])
        hour, minute = map(int, input["time"].split(":"))
        time_index = hour * 60 + minute
        self.plot_tab.update_data(satelites_epoch, input["el_mask"])

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