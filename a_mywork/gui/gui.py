import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, QtCore
import cv2

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, Worker):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 960)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.cam1_frame = QtWidgets.QLabel(self.centralwidget)
        self.cam1_frame.setGeometry(QtCore.QRect(0, 10, 960, 720))
        self.cam1_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.cam1_frame.setObjectName("cam1_frame")
        self.cam1_frame.setStyleSheet("border: 1px solid blue;")
        self.cam2_frame = QtWidgets.QLabel(self.centralwidget)
        self.cam2_frame.setGeometry(QtCore.QRect(960, 10, 960, 720))
        self.cam2_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.cam2_frame.setObjectName("cam2_frame")
        self.cam2_frame.setStyleSheet("border: 1px solid blue;")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(0, 740, 1920, 220))
        self.textBrowser.setObjectName("textBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.Worker = Worker

        self.Worker.start()
        self.Worker.ImageUpdate1.connect(self.ImageUpdateSlot1)
        self.Worker.ImageUpdate2.connect(self.ImageUpdateSlot2)
        self.Worker.InfoUpdate.connect(self.InfoUpdateSlot)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def ImageUpdateSlot1(self, Image):
        self.cam1_frame.setPixmap(QPixmap.fromImage(Image))

    def ImageUpdateSlot2(self, Image):
        self.cam2_frame.setPixmap(QPixmap.fromImage(Image))

    def InfoUpdateSlot(self, mess):
        self.textBrowser.append(mess+'\n')

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("The best GUI ever created")
        MainWindow.setFixedWidth(1920)
        MainWindow.setFixedHeight(960)
        self.cam1_frame.setText("No signal from cam 1")
        self.cam2_frame.setText("No signal from cam 2")
    