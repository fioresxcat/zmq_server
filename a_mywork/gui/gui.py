import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, QtCore
import cv2
import threading


class thread(threading.Thread):
    def __init__(self, thread_name, thread_ID):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.thread_ID = thread_ID
    def run(self):
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.cam1_frame = QtWidgets.QLabel(self.centralwidget)
        self.cam1_frame.setGeometry(QtCore.QRect(0, 50, 640, 480))
        self.cam1_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.cam1_frame.setObjectName("cam1_frame")
        self.cam1_frame.setStyleSheet("border: 1px solid blue;")
        self.cam2_frame = QtWidgets.QLabel(self.centralwidget)
        self.cam2_frame.setGeometry(QtCore.QRect(640, 50, 640, 480))
        self.cam2_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.cam2_frame.setObjectName("cam2_frame")
        self.cam2_frame.setStyleSheet("border: 1px solid blue;")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(0, 550, 1280, 231))
        self.textBrowser.setObjectName("textBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.Worker = Worker()

        self.Worker.start()
        self.Worker.ImageUpdate.connect(self.ImageUpdateSlot1)
        self.Worker.ImageUpdate.connect(self.ImageUpdateSlot2)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def ImageUpdateSlot1(self, Image):
        self.cam1_frame.setPixmap(QPixmap.fromImage(Image))

    def ImageUpdateSlot2(self, Image):
        self.cam2_frame.setPixmap(QPixmap.fromImage(Image))

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("The best GUI ever created")
        MainWindow.setFixedWidth(1280)
        MainWindow.setFixedHeight(800)
        self.cam1_frame.setText("No signal from cam 1")
        self.cam2_frame.setText("No signal from cam 2")


class Worker(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    def stop(self):
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
    