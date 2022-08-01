from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog

from TrafficanteView import Ui_MainWindow
from TrafficanteModel import TrafficanteModel
import sys
import os
import shutil
from pathlib import Path


from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np
import  zmq

class TrafficanteApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(TrafficanteApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.trafficante = TrafficanteModel()

        self.ui.symbolNameBox.addItems( self.trafficante.pairSettings.availablePairs )
        self.ui.availableTimeframes.addItems( self.trafficante.pairSettings.availablePeriods )

        self.onlineUpdateFlag = self.ui.onlineUpdate.isChecked()

        self.plotCanvas = PlotCanvas(self.ui.plotAreaWidget)
        self.ui.updatePlotButton.clicked.connect( self.updatePlot )

    def updatePlot(self):

        self.setSettings()

        self.onlineUpdateFlag = self.ui.onlineUpdate.isChecked()

        if self.onlineUpdateFlag:
            self.onlineUpdate()
        else:
            self.offlineUpdate()

        self.plotCanvas.plot( self.trafficante )

    def offlineUpdate(self):

        fileDialog = QFileDialog()
        chosedFile = fileDialog.getOpenFileName(self)
        chosedFile = chosedFile[0]

        self.trafficante.updateBarData( chosedFile )

        pass

    def onlineUpdate(self):

        zmqContext = zmq.Context()
        socket = zmqContext.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")

        reqString = self.trafficante.pairSettings.pairName + "_" + self.trafficante.pairSettings.pairPeriod
        socket.send_string(reqString)

        downloadedDataPath = socket.recv()
        downloadedDataPath = downloadedDataPath.decode()

        self.trafficante.updateBarData( downloadedDataPath )

        pass

    def setSettings(self):

        settings = self.trafficante.pairSettings
        settings.pairName = self.ui.symbolNameBox.currentData(0)
        settings.pairPeriod = self.ui.availableTimeframes.currentData(0)

        self.trafficante.setSettings( settings )


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)

        self.setParent(parent)

        self.toolbar = NavigationToolbar2QT(self, self)
        self.toolbar.setGeometry(QtCore.QRect(150, 20, 611, 40))
        self.toolbar.show()

        FigureCanvas.updateGeometry(self)

    def plot(self, trafficanteObject):

        ax = self.axes
        ax.cla()

        plotData = trafficanteObject.getPredictPlotData()

        for i in range( len( plotData ) ):
            curPlotData = plotData[i]

            for key in curPlotData.keys():
                Y_set = curPlotData[key]

                y_axis = np.zeros((Y_set.shape[1], ))
                for j in range (Y_set.shape[1]):
                    y_axis[j] = Y_set[0][j]

                x_axis = np.zeros((y_axis.shape[0], ))
                for j in range( x_axis.shape[0] ):
                    x_axis[j] = j

                ax.plot(x_axis, y_axis, label=key)

        ax.legend()
        self.draw()

        FigureCanvas.updateGeometry(self)

        return True


app = QtWidgets.QApplication([])
application = TrafficanteApp()
application.show()

sys.exit(app.exec())