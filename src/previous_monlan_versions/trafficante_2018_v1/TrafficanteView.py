# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\BookKnight\Desktop\gui.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(881, 721)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(610, 0, 271, 511))
        self.tabWidget.setObjectName("tabWidget")
        self.usingTab = QtWidgets.QWidget()
        self.usingTab.setObjectName("usingTab")
        self.updatePlotButton = QtWidgets.QPushButton(self.usingTab)
        self.updatePlotButton.setGeometry(QtCore.QRect(10, 170, 121, 23))
        self.updatePlotButton.setObjectName("updatePlotButton")
        self.symbolNameBox = QtWidgets.QComboBox(self.usingTab)
        self.symbolNameBox.setGeometry(QtCore.QRect(90, 30, 69, 22))
        self.symbolNameBox.setObjectName("symbolNameBox")
        self.availableTimeframes = QtWidgets.QComboBox(self.usingTab)
        self.availableTimeframes.setGeometry(QtCore.QRect(90, 70, 71, 22))
        self.availableTimeframes.setObjectName("symbolNameBox_2")
        self.symbolNameLabel = QtWidgets.QLabel(self.usingTab)
        self.symbolNameLabel.setGeometry(QtCore.QRect(10, 30, 81, 21))
        self.symbolNameLabel.setObjectName("symbolNameLabel")
        self.periodLabel = QtWidgets.QLabel(self.usingTab)
        self.periodLabel.setGeometry(QtCore.QRect(10, 70, 61, 16))
        self.periodLabel.setObjectName("periodLabel")
        self.onlineUpdate = QtWidgets.QRadioButton(self.usingTab)
        self.onlineUpdate.setGeometry(QtCore.QRect(10, 140, 91, 17))
        self.onlineUpdate.setObjectName("onlineUpdate")
        self.tabWidget.addTab(self.usingTab, "")
        self.trainingTab = QtWidgets.QWidget()
        self.trainingTab.setObjectName("trainingTab")
        self.tabWidget.addTab(self.trainingTab, "")
        self.plotAreaWidget = QtWidgets.QWidget(self.centralwidget)
        self.plotAreaWidget.setGeometry(QtCore.QRect(0, 0, 611, 511))
        self.plotAreaWidget.setAutoFillBackground(False)
        self.plotAreaWidget.setObjectName("plotAreaWidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(0, 510, 881, 171))
        self.textBrowser.setObjectName("textBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 881, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.updatePlotButton.setText(_translate("MainWindow", "Update plot"))
        self.symbolNameLabel.setText(_translate("MainWindow", "Symbol name"))
        self.periodLabel.setText(_translate("MainWindow", "Period"))
        self.onlineUpdate.setText(_translate("MainWindow", "Online update"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.usingTab), _translate("MainWindow", "Using"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.trainingTab), _translate("MainWindow", "Training"))

