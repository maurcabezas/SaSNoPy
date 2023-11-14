from PyQt5.QtWidgets import QApplication, QComboBox,QMainWindow, QWidget,QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtWidgets import QLineEdit, QSlider, QLabel,QMessageBox, QRadioButton, QLineEdit,QDesktopWidget
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PyQt5.QtGui import QIntValidator
import os, sys
import pandas as pd
from scipy.interpolate import UnivariateSpline
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.special import legendre
from scipy.stats import norm
# from scipy.special.orthogonal import Legendre
import argparse
from numpy.polynomial import Legendre, Chebyshev



class SpectrumNormalizer(QMainWindow):
    def __init__(self, filename,is_list=False):
        super().__init__()
        self.filename = filename
        self.file_list = []
        self.current_file_index = 0
        self.is_list = is_list
        self.initUI()
        self.show_welcome_message()
        # print ('IS LIST:',is_list)
        if is_list:
            self.read_spectrum_list(filename)
        else:
            self.read_spectrum(filename)
            self.next_button.setEnabled(False)
            self.prev_button.setEnabled(False)


    def initUI(self):
        bsh=30
        bsw=150
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QHBoxLayout(self.central_widget)

        canvas_layout=QVBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        canvas_layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(211)
        self.ax.set_xlabel('Wavelength')
        self.ax.set_ylabel('Flux')

        self.ax1 = self.figure.add_subplot(212)
        self.ax1.set_xlabel('Wavelength')
        self.ax1.set_ylabel('Normalized Flux')
        # canvas_layout.addStretch()
        layout.addLayout(canvas_layout)
# ##############################
        bottom_layout=QHBoxLayout()
        self.label1 = QLabel('Input file:')
        self.label1.setFixedSize(bsw, 20)
        # layout.addLayout(canvas_layout)
        bottom_layout.addWidget(self.label1)
        bottom_layout.addStretch()
        bottom_layout.setGeometry(QtCore.QRect(10, 10, 10, 10))
        bottom_layout.setContentsMargins(0,0, 0, 0)
        # canvas_layout.setSpacing(10)
        canvas_layout.addLayout(bottom_layout)
################################
        # canvas_layout.addStretch()
        button_layout = QVBoxLayout()  # Vertical layout for buttons

        self.clear_button = QPushButton('Clear Selection')
        self.clear_button.clicked.connect(self.clear_selection)
        self.clear_button.setEnabled(False)
        self.clear_button.setFixedSize(bsw, bsh)
        button_layout.addWidget(self.clear_button)

        # self.normalize_button = QPushButton('Normalize Spectrum')
        # self.normalize_button.clicked.connect(self.normalize_spectrum)
        # self.normalize_button.setEnabled(False)
        # layout.addWidget(self.normalize_button)



        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.setEnabled(True)
        self.reset_button.setFixedSize(bsw, bsh)
        button_layout.addWidget(self.reset_button)

        # Add a combo box for fit function selection
        self.fit_function_combo = QComboBox()
        self.fit_function_combo.addItems(['Function: Spline3', 'Function: Legendre', 'Function: Chebyshev'])
        self.fit_function_combo.setCurrentIndex(0)  # Set 'spline3' as default
        self.selected_fit_function = 'spline3'  # Default fit function
        self.fit_function_combo.currentIndexChanged.connect(self.update_fit_function)
        self.fit_function_combo.setFixedSize(bsw,bsh)
        button_layout.addWidget(self.fit_function_combo)

        # Add a combo box for order of function selection
        self.order_function_combo = QComboBox()
        self.order_function_combo.addItems(['order: 1', 'order: 2','order: 3','order: 4','order: 5'])
        self.order_function_combo.setCurrentIndex(2)  # Set 'spline3' as default
        self.selected_order_function = '3'  # Default fit function
        self.order_function_combo.currentIndexChanged.connect(self.update_order_function_combo)
        self.order_function_combo.setFixedSize(bsw,bsh)
        button_layout.addWidget(self.order_function_combo)


##############################
        self.divide_spec_button = QPushButton('Divide spectrum')
        self.divide_spec_button.clicked.connect(self.divide_spectrum)
        self.divide_spec_button.setEnabled(True)
        self.divide_spec_button.setFixedSize(bsw, bsh)
        button_layout.addWidget(self.divide_spec_button)

        sub_layout1 = QHBoxLayout()
        self.divide_spec_button1 = QPushButton('<')
        self.divide_spec_button2 = QPushButton('>')
        # self.divide_spec_button.clicked.connect(self.reset)
        self.divide_spec_button1.setEnabled(False)
        self.divide_spec_button2.setEnabled(False)
        self.divide_spec_button1.setFixedSize(bsw/2, bsh)
        self.divide_spec_button2.setFixedSize(bsw/2, bsh)
        sub_layout1.addWidget(self.divide_spec_button1)
        sub_layout1.addWidget(self.divide_spec_button2)
        button_layout.addLayout(sub_layout1)

################################
        button_layout.addStretch()

        # Add arrow buttons for navigating through the list
        self.prev_button = QPushButton('Previous spectrum')
        self.prev_button.clicked.connect(self.load_previous_spectrum)
        self.prev_button.setFixedSize(bsw, bsh)
        button_layout.addWidget(self.prev_button)

        self.next_button = QPushButton('Next spectrum')
        self.next_button.clicked.connect(self.load_next_spectrum)
        self.next_button.setFixedSize(bsw, bsh)
        button_layout.addWidget(self.next_button)

        # Add radio button for autosave prefix
        self.autosave_label = QLabel('Autosave Prefix:')
        self.autosave_radio_yes = QRadioButton('Yes')
        self.autosave_radio_no = QRadioButton('No')
        self.autosave_textbox = QLineEdit()
        self.autosave_textbox.setEnabled(False)
        self.autosave_textbox.setFixedSize(bsw,bsh)

        # Set 'No' as the default state
        self.autosave_radio_no.setChecked(True)

        # Connect the stateChanged signal of the radio button to a custom method
        self.autosave_radio_yes.toggled.connect(self.toggle_autosave_textbox)

        autosave_layout = QVBoxLayout()
        autosave_layout.addWidget(self.autosave_label)
        autosave_layout.addWidget(self.autosave_radio_yes)
        autosave_layout.addWidget(self.autosave_radio_no)
        autosave_layout.addWidget(self.autosave_textbox)
        autosave_layout.addStretch()

        # Add autosave layout to the existing button_layout
        button_layout.addLayout(autosave_layout)                

        self.save_button = QPushButton('Save\nNormalized Spectrum')
        self.save_button.clicked.connect(self.save_normalized_spectrum)
        self.save_button.setEnabled(False)
        self.save_button.setFixedSize(bsw, 2*bsh)
        button_layout.addWidget(self.save_button)

         # Add a Quit button
        self.quit_button = QPushButton('Quit')
        self.quit_button.clicked.connect(self.confirm_quit)
        self.quit_button.setFixedSize(bsw, bsh)
        button_layout.addWidget(self.quit_button)

        layout.addLayout(button_layout)
########################################################
        self.selected_indices = []
        self.selected_points = []
        self.spectrum_loaded = False
        self.spectrum = None
        #self.continuum = None
        self.lines = None  # To store line segments

        # Create a connection for the 'button_press_event' event (left-click)
        self.canvas.mpl_connect('button_press_event', self.onclick)

        self.central_widget.setFocusPolicy(Qt.StrongFocus)  # Set focus policy to enable keyboard events
        self.central_widget.keyPressEvent = self.keyPressEvent  # Connect key press event to custom method
 
    def divide_spectrum(self):
        # Implement the logic to divide the spectrum here
        # You can use the QMessageBox to inform the user about the division
        total_range = self.wave[-1] - self.wave[0]
    
        # Set the desired region size
        region_size_limit = 199
        overlap=10
        file_format = '%02d_%s.tmp'

        # Calculate the number of regions
        num_regions = int(total_range / region_size_limit)
        print (total_range / region_size_limit)
        actual_region_size=round(total_range / num_regions, 1)
        print (actual_region_size)

        # Inform the user about the division
        QMessageBox.information(self, 'Spectrum Divided', f'The spectrum has been divided into {num_regions} regions of {actual_region_size} (+{overlap} overlap) angstroms each.')

        temp_files = []

        for i in range(1, num_regions + 1):
            print (i)
            if i==1:
                initial_wave = self.wave[0] + (i - 1) * actual_region_size
                final_wave = initial_wave + actual_region_size + overlap
            elif i==(num_regions):
                print (i)
                initial_wave = self.wave[0] + (i - 1) * actual_region_size
                final_wave = initial_wave + actual_region_size 
            else:
                initial_wave = self.wave[0] + (i - 1) * actual_region_size
                final_wave = initial_wave + actual_region_size + overlap               

            initial_wave=round(initial_wave)
            final_wave=round(final_wave)

            # Use initial_wave and final_wave for further processing or display
            print(f"Region {i}: Initial Wave - {initial_wave}, Final Wave - {final_wave}")

            # Extract the spectrum for the current region
            region_spectrum = self.spectrum[(self.wave >= initial_wave) & (self.wave <= final_wave)]

            # Save the region spectrum to a temporary file
            temp_file_path = file_format % (i, self.specfile)

            # np.savetxt(temp_file_path, np.column_stack([self.wave[(self.wave >= initial_wave) & (self.wave <= final_wave)], region_spectrum]))
            np.savetxt(temp_file_path, region_spectrum, fmt='%10.6f %10.6f', delimiter='\t')

            temp_files.append(temp_file_path)

        list_file_path = '%s_list.tmp' % self.specfile
        with open(list_file_path, 'w') as list_file:
            for temp_file_path in temp_files:
                list_file.write(temp_file_path + '\n')

        self.read_spectrum_list(list_file_path)

        self.divide_spec_button1.setEnabled(True)
        self.divide_spec_button2.setEnabled(True)
        self.divide_spec_button.setEnabled(False)


        # Add logic to actually divide the spectrum
        # ...

    def show_welcome_message(self):
        welcome_box = QMessageBox(self)
        welcome_box.setWindowTitle('Welcome')
        welcome_box.setText('To enhance visualization, consider dividing wide spectra into regions of 250 angstroms.')
        # Add any additional help or indications here

        # Center the QMessageBox on the screen
        screen_geometry = QDesktopWidget().screenGeometry()
        welcome_box.move((screen_geometry.width() - welcome_box.width()) // 2, (screen_geometry.height() - welcome_box.height()) // 2)

        welcome_box.exec_()

    def toggle_autosave_textbox(self, state):
        # Enable/disable the autosave textbox based on the state of the radio button
        if state==True:
            self.autosave_textbox.setEnabled(True)
        elif state==False:
            self.autosave_textbox.setEnabled(False)

    def load_previous_spectrum(self):
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.clear_selection()  # Clear the selection points
            self.ax.clear()
            self.read_spectrum(self.file_list[self.current_file_index])
            self.ax1.clear()
            self.reset()

    def load_next_spectrum(self):
        if self.current_file_index < len(self.file_list) - 1:
            self.current_file_index += 1
            self.ax.clear()
            self.read_spectrum(self.file_list[self.current_file_index])
            self.ax1.clear()
            self.reset()


    def confirm_quit(self):
        reply = QMessageBox.question(self, 'Confirmation', 'Are you sure you want to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.close()

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_C:
            self.clear_selection()
        elif key == Qt.Key_R:
            self.reset()
        elif key == Qt.Key_S:
            self.save_normalized_spectrum()
        elif key == Qt.Key_Q:
            self.confirm_quit()
        elif key == Qt.Key_Left:
            self.load_previous_spectrum()
        elif key == Qt.Key_Right:
            self.load_next_spectrum()
        elif key == Qt.Key_Space:
            # Perform the actions of both save_normalized_spectrum and load_next_spectrum
            self.save_normalized_spectrum()
            self.load_next_spectrum()

    def read_spectrum_list(self, list_filename):
        # print(self.is_list)
        with open(list_filename, 'r') as file:
            self.file_list = [line.strip() for line in file if line.strip()]

        if self.file_list:
            self.read_spectrum(self.file_list[0])

    def read_spectrum(self, filename):
        # print ('read_spectrum function')
        # print (filename)
        specfile, file_extension = os.path.splitext(filename)
        if file_extension in (".fits", ".fit"):
            inspec = filename
            print('Loaded spectrum:',inspec)
            sp = fits.open(inspec)
            hdr = sp[0].header
            flux = sp[0].data
            w = WCS(hdr, naxis=1, relax=False, fix=False)
            wave = w.wcs_pix2world(np.arange(len(flux)), 0)[0]
        else:
            inspec = filename
            print('Loaded spectrum:',inspec)
            spec = pd.read_csv(inspec, delim_whitespace=True, header=None)
            wave = np.array(spec[0])
            flux = np.array(spec[1])

        if wave[-1]-wave[0] > 230:
            self.divide_spec_button.setEnabled(True)
        else:
            self.divide_spec_button.setEnabled(False)
        self.specfile=specfile
        self.load_spectrum=filename
        self.wave, self.flux = wave, flux
        self.rr=0
        # Set self.spectrum here
        self.spectrum = np.column_stack((wave, flux))
        self.snr_spectrum()
        self.ax.clear()
        self.get_continuum_points()
        self.continuum(self.wave_points,self.flux_points)
        self.selected_indices,self.selected_points=self.wave_points,self.flux_points
        self.ax.plot(self.wave_points, self.flux_points, 'or')
        self.ax.plot(self.wave, self.flux, '-k',zorder=0)
        self.ax.set_xlabel('Wavelength')
        self.ax.set_ylabel('Flux')

        self.clear_button.setEnabled(True)

        if self.is_list:
            self.setWindowTitle(f'Spectrum Normalizer - {filename} ({self.current_file_index + 1}/{len(self.file_list)})')
        else:
            self.setWindowTitle('Spectrum Normalizer')


    def onclick(self, event):
        # When the left mouse button is clicked, add a continuum point (red square marker)
        # closest_index = np.argmin(np.abs(self.selected_indices - event.xdata))
        if event.xdata is not None and event.ydata is not None:
            if event.button == 1:
                self.clear_button.setEnabled(True)
                # print('onclick function')
                self.selected_indices = np.append(self.selected_indices, event.xdata)  # Append the new point
                self.selected_points = np.append(self.selected_points, event.ydata)  # Append the new point
                # print(self.selected_indices, self.selected_points)
                self.ax.plot(event.xdata, event.ydata, 'or')
                self.reset_button.setEnabled(True)
                print ('Added point:',event.xdata,event.ydata)
                self.plot_spectrum()
                self.canvas.draw()
                # print ('SELECTED POINTS:',len(self.selected_indices))
                # print ('USING RESET->',self.rr)
                if self.rr==1:
                    # print ('USING RESET->',self.rr)
                    self.ax.plot(self.wave, self.flux, '-k',zorder=0)
                    self.canvas.draw()
                #     self.rr=0
                if len(self.selected_indices) > 5:
                    self.continuum(self.selected_indices, self.selected_points)
                self.rr=0
            elif event.button == 3:  # Right-click
                # print('delete function')
                closest_index = np.argmin(np.abs(self.selected_indices - event.xdata))
                if np.abs(self.selected_indices[closest_index] - event.xdata) < 0.5:
                    # print (self.selected_indices,closest_index)
                    self.selected_indices = np.delete(self.selected_indices, closest_index)
                    self.selected_points = np.delete(self.selected_points, closest_index)
                    self.spectrum = np.column_stack((self.selected_indices, self.selected_points))
                    self.clear_plot()  # Clear the plot
                    # self.draw_connection_lines()
                    self.plot_spectrum()
                    self.reset_button.setEnabled(True)  # Re-plot with updated points
                    print ('Deleted point:',event.xdata,event.ydata)
                    if len(self.selected_indices) > 5:
                        self.continuum(self.selected_indices, self.selected_points)

    def clear_plot(self):
        # print('clear plot function')
        self.ax.clear()
        # self.ax.plot(self.wave, self.normalized_flux, '-b',zorder=0)
        self.ax.set_xlabel('Wavelength')
        self.ax.set_ylabel('Flux')
        self.ax.plot(self.wave, self.flux, '-k',zorder=0)
        self.canvas.draw()
        # self.normalize_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.ax1.clear()

    def clear_selection(self):
        # print('clear_selection function')
        self.selected_indices = []
        self.selected_points = []
        self.clear_plot()
        self.clear_button.setEnabled(False)
        self.reset_button.setEnabled(True)
    def plot_spectrum(self):
        # print ('plot_spectrum function')
        if self.spectrum is not None:
            self.ax.plot(self.selected_indices, self.selected_points, 'or',  picker=5)
            self.ax.set_xlabel('Wavelength')
            self.ax.set_ylabel('Flux')
            self.canvas.draw()

    def plot_input_spectrum(self):
        # print ('plot_input_spectrum function')
        if self.spectrum is not None:
            #self.ax.clear()
            self.ax.plot(self.wave, self.flux, '-k')
            self.ax.set_xlabel('Wavelength')
            self.ax.set_ylabel('Flux')
            self.canvas.draw()

    # def draw_connection_lines(self):
    #     if self.lines is not None:
    #         for line in self.lines:
    #             line.remove()
    #     if len(self.selected_indices) > 1:
    #         sorted_indices, sorted_points = zip(*sorted(zip(self.selected_indices, self.selected_points)))
    #         self.lines = self.ax.plot(sorted_indices, sorted_points, 'r')
    #     else:
    #         self.lines = None

    def normalize_spectrum(self):
        # print('normalize_spectrum function')
        self.ax.clear()
        self.ax.plot(self.spectrum[:, 0], self.spectrum[:, 1], '-k')
        self.ax.set_xlabel('Wavelength')
        self.ax.set_ylabel('Flux')

        self.normalized_flux = np.interp(self.spectrum[:, 0], [xmin for xmin, _ in self.selected_indices], [y for _, y in self.selected_points])
        normalized_spectrum = self.spectrum[:, 1] / self.continuum

        self.ax.plot(self.spectrum[:, 0], normalized_spectrum, '-r')
        self.canvas.draw()

        self.clear_button.setEnabled(False)
        # self.normalize_button.setEnabled(False)
        self.save_button.setEnabled(True)

    def save_normalized_spectrum(self):
        # print('save_normalize_spectrum function')

        # Check the state of the autosave radio button
        if self.autosave_radio_yes.isChecked():
            # Get the prefix from the textbox
            prefix = self.autosave_textbox.text()

            if not prefix:
                print("Error: Please enter a prefix for autosave.")
                return

            # Get the directory of the input spectrum
            directory = os.path.dirname(self.filename)

            # Construct the autosave filename
            autosave_filename = os.path.join(directory, f"{prefix}{os.path.basename(self.load_spectrum)}")

            # Save the normalized spectrum using the autosave filename
            np.savetxt(autosave_filename, np.column_stack((self.wave, self.flux / self.normalized_flux)))

            print(f"Autosaved normalized spectrum as: {autosave_filename}")

        else:
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save Normalized Spectrum', '', 'All Files (*)', options=options)

            if file_path:
                # Save the normalized spectrum using self.wave and self.normalized_flux
                np.savetxt(file_path, np.column_stack((self.wave, self.flux/self.normalized_flux)))

    def snr_spectrum(self):
        """
        Calculate Signal-to-Noise Ratio (SNR) of the input spectrum.
        """
        # if self.spectrum is None:
        #     print("Error: No spectrum loaded.")
        #     return

        # # Assuming the noise is the standard deviation of the flux
        # noise = np.std(self.spectrum[:, 1])

        # # Calculate SNR
        # snr = np.median(self.spectrum[:, 1]) / noise

        # self.snr = snr
        # print(f"SNR of the spectrum: {snr}")

        if self.spectrum is None:
            print("Error: No spectrum loaded.")
            return
        # Define the wavelength range for background estimation
        background_min_wavelength = self.wave[0]
        background_max_wavelength = self.wave[-1]

        # Assuming the noise is the standard deviation of the background
        # You can specify a specific wavelength range for background estimation
        background_flux = self.flux[(self.wave < background_max_wavelength) & (self.wave > background_min_wavelength)]
        noise = np.std(background_flux)

        # Calculate SNR
        snr = np.median(self.spectrum[:, 1]) / noise

        self.snr = snr
        print(f"SNR of the spectrum: {snr}")

    def reset(self):
        self.rr=1
        self.clear_selection()
        #self.spectrum_loaded = False
        #self.spectrum = None
        # self.continuum = None
        self.selected_indices = self.wave_points.copy()
        self.selected_points = self.flux_points.copy()
        self.ax.clear()
        # self.get_continuum_points()
        # self.plot_input_spectrum()
        self.ax.plot(self.wave, self.flux, '-k', zorder=0)
        self.ax.plot(self.selected_indices, self.selected_points, 'or')
        self.ax.set_xlabel('Wavelength')
        self.ax.set_ylabel('Flux')
        self.continuum(self.wave_points,self.flux_points)
        # self.ax1.clear()
        self.ax.plot(self.wave, self.normalized_flux, '-b',zorder=0)
        #self.continuum(self.wave, self.flux)
        self.canvas.draw()

        # self.load_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        # self.normalize_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.reset_button.setEnabled(False)

    def get_continuum_points(self, top=50, eqw_threshold=10, absorption_range=None):
        # print('get_continuum function')
        """Get continuum points along a spectrum.

        This splits a spectrum into "splits" number of bins and calculates
        the median wavelength and flux of the upper "top" number of flux
        values.
        """
        self.top=top
        splits=20
        wave = self.wave
        flux = self.flux

        # Shorten array until it can be evenly split up.
        remainder = len(flux) % splits
        if remainder:
            # Non-zero remainder needs this slicing
            wave = wave[:-remainder]
            flux = flux[:-remainder]

        wave_shaped = wave.reshape((splits, -1))
        flux_shaped = flux.reshape((splits, -1))
        # print ('TOP HI REJ:',self.top)
        s = np.argsort(flux_shaped, axis=-1)[:, -self.top:]

        s_flux = np.array([ar1[s1] for ar1, s1 in zip(flux_shaped, s)])
        s_wave = np.array([ar1[s1] for ar1, s1 in zip(wave_shaped, s)])

        wave_points = np.median(s_wave, axis=-1)
        flux_points = np.median(s_flux, axis=-1)
        assert len(flux_points) == splits

        self.wave_points = wave_points
        self.flux_points = flux_points
        # ################
        # # Calculate equivalent width for each split
        # eqw_values = self.calculate_eqw(s_wave, s_flux)

        # # Ensure eqw_values is a 2D array
        # eqw_values = np.atleast_2d(eqw_values)

        # # Check conditions for each spectrum split separately
        # valid_points_mask = ~self.is_within_range(s_wave, absorption_range) & (np.all(eqw_values >= eqw_threshold, axis=1))

        # # Check if any valid points exist
        # if np.any(valid_points_mask):
        #     # Use the mask to filter out invalid points
        #     wave_points = np.median(s_wave[valid_points_mask], axis=-1)
        #     flux_points = np.median(s_flux[valid_points_mask], axis=-1)
        # else:
        #     # Set wave_points and flux_points to NaN if no valid points
        #     wave_points = np.full(splits, np.nan)
        #     flux_points = np.full(splits, np.nan)

        # self.wave_points = wave_points
        # self.flux_points = flux_points



    # def is_within_range(self, wave, absorption_range):
    #     """
    #     Check if wavelength points are within the specified range.
    #     """
    #     if absorption_range is None:
    #         return np.full_like(wave, True, dtype=bool)
    #     else:
    #         return np.logical_or(wave < absorption_range[0], wave > absorption_range[1])

    # def calculate_eqw(self, wave, flux):
    #     """
    #     Calculate equivalent width (EQW) for each split.
    #     """
    #     # Calculate EQW using your preferred method
    #     eqw_values = np.trapz(1.0 - flux, wave, axis=-1)

    #     # Ensure eqw_values is a 1D array
    #     eqw_values = np.atleast_1d(eqw_values)

    #     return eqw_values

    def update_fit_function(self, index):
        # Update the fit function based on the selected option
        fit_functions = ['spline3', 'legendre','chebyshev']
        self.selected_fit_function = fit_functions[index]
        # print (self.selected_fit_function)
        # self.continuum()
        # Update the continuum function
        if len(self.selected_indices) > 5:
            self.continuum(self.selected_indices, self.selected_points)
            self.plot_spectrum()
            self.canvas.draw()

    def update_order_function_combo(self, index):
        # Update the fit function based on the selected option
        order_fit_functions = ['1','2','3','4','5']
        self.selected_order_function = int(order_fit_functions[index])
        # print (self.selected_order_function)
        # self.continuum()
        # Update the continuum function
        if len(self.selected_indices) > 5:
            self.continuum(self.selected_indices, self.selected_points)
            self.plot_spectrum()
            self.canvas.draw()

    def continuum(self,nwave,nflux):
        # print('continuum function')
        """Fit continuum of flux.
        
        top: is the number of top points to take the median of the continuum.
        """
        function=self.selected_fit_function #'spline3'   #poly, legendre, spline3
        order = int(self.selected_order_function)
        # top=20
        # splits=25
        wave = self.wave
        flux = self.flux

        org_wave = wave[:]
        org_flux = flux[:]

        # Get continuum value in chunked sections of the spectrum.

        wave_points, flux_points = nwave, nflux
    # Ensure wave_points are sorted in ascending order
        sorted_indices = np.argsort(wave_points)
        wave_points = wave_points[sorted_indices]
        flux_points = flux_points[sorted_indices]
        # wave_points, flux_points = self.selected_indices, self.selected_points
        # print ('Continuum ORDER:',self.selected_order_function)
        # print (wave_points,flux_points)

        # poly_num = {"scalar": 0, "linear": 1, "quadratic": 2, "cubic": 3}

        if function == 'spline3':
            # Fit a cubic spline to the continuum points
            # sorted_indices = np.argsort(wave_points)
            # sorted_wave_points = wave_points[sorted_indices]
            # sorted_flux_points = flux_points[sorted_indices]
            # spline = UnivariateSpline(sorted_wave_points, sorted_flux_points, k=order, s=0.0005)

            spline = UnivariateSpline(wave_points, flux_points, k=order, s=0.0005)
            # Evaluate the spline to obtain the normalized flux
            norm_flux = spline(org_wave)
            self.normalized_flux = norm_flux
        elif function == 'chebyshev':
            # Fit a Chebyshev polynomial to the continuum points
            coeff = Chebyshev.fit(wave_points, flux_points, order)
            norm_flux = coeff(org_wave)
            self.normalized_flux = norm_flux
        elif function == 'legendre':
            # Fit a Legendre polynomial to the continuum points
            # max_order = min(max_order, len(wave_points) - 1)
            coeff = Legendre.fit(wave_points, flux_points, order)
            norm_flux = coeff(org_wave)
            self.normalized_flux = norm_flux
        # self.normalized_flux = norm_flux

        # print ('FLUX-NORM:',self.normalized_flux)
        # print ('FLUX-NORM:',self.wave)
        self.ax.clear()
        self.plot_input_spectrum()
        self.ax.plot(self.selected_indices, self.selected_points, 'or',  picker=5)
        self.ax.plot(self.wave, self.normalized_flux, '-b',zorder=0)
        
        self.ax1.clear()
        self.ax1.axhline(y=1, color='red', linestyle='--')
        self.ax1.plot(self.wave, flux/self.normalized_flux, '-k',zorder=0)
        self.ax1.set_xlabel('Wavelength')
        self.ax1.set_ylabel('Normalized Flux')
        self.ax1.set_xlim(self.ax.get_xlim())
        self.canvas.draw()
        self.save_button.setEnabled(True)


def main():
    parser = argparse.ArgumentParser(description='Spectrum Normalizer')
    parser.add_argument('filename', help='Path to the spectrum file')
    parser.add_argument('-list', '-l', action='store_true', help='Indicate that the filename is a list of spectra')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = SpectrumNormalizer(args.filename, args.list)  # Pass the filename as an argument
    window.setWindowTitle('Spectrum Normalizer')
    window.setGeometry(100, 100, 1400, 700)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

