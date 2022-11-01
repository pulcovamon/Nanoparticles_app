from functools import partial
import sys
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QFormLayout,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QSpinBox
)
from PyQt6.QtGui import (
    QPixmap
)
from PyQt6.QtCore import (

)
import image_processing_functions
from gettext import find
from operator import index
import cv2
from cv2 import threshold
import numpy as np

''' 
Function for create GUI window.
()
Parameters:
----
'''

class Window(QMainWindow):

    def __init__(self):
        super().__init__(parent=None)

        self.setWindowTitle('Nanoparticles app')

        layout = QVBoxLayout()

        status_text = 'No image added'
        self._createToolBar()
        self._createStatusBar(status_text)
        #self._defineGeometry()
        
        min_value = 10
        max_value = 100
        my_text = 'Insert size of NPs'
        self._createButton(layout, min_value, max_value, my_text)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

    def _createMenu(self):
        menu = self.menuBar().addMenu('&Menu')
        menu.addAction("&Exit", self.close)
        menu.addAction('Add new image')

    def _createToolBar(self):
        tools = QToolBar()
        tools.addAction('Start')
        tools.addAction('Raw image')
        tools.addAction('Blured image')
        tools.addAction('Binarized image')
        tools.addAction("Exit", self.close)
        self.addToolBar(tools)

    def _createStatusBar(self, status_text):
        status = QStatusBar()
        status.showMessage(status_text)
        self.setStatusBar(status)

    def _defineGeometry(self):
        self.setGeometry(200, 200, 800, 400)

    def _createButton(self, layout, min_value, max_value, my_text):
        button = QPushButton("Add new image")
        button.clicked.connect(lambda: self._addImage(layout))
        layout.addWidget(button)

        combo = QComboBox()
        combo.addItems(['Nanoparticles', 'Nanorods'])
        combo.currentIndexChanged.connect(partial(self._index_changed, my_layout = layout))
        layout.addWidget(combo)

        my_label = QLabel(my_text)
        layout.addWidget(my_label)
        
        number = QSpinBox()
        number.setMinimum(min_value)
        number.setMaximum(max_value)
        number.setSingleStep(10)
        layout.addWidget(number)

    def _index_changed(self, index, my_layout):
        if index == 0:
            my_text = 'Insert size of AuNPs'
            min_value = 10
            max_value = 100
        else:
            my_text = 'Insert peak wavelength of GNRs'
            min_value = 400
            max_value = 1500
        self._createButton(my_layout, min_value, max_value, my_text)

    def _addImage(self, layout):
        image_processing_functions.loadImg()
        status_text = '1 image added'
        self._createStatusBar(status_text)

        label = QLabel(self)
        pixmap = QPixmap('AuNP50nm_008.jpg')
        pixmap = pixmap.scaled(250, 250, QtCore.Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
        layout.addWidget(label)
        self.setLayout(layout)


    def _rawImage(self, img, widget, layout):
        widget.setPixmap(QPixmap(img))
        layout.removeWidget(self.button)
        layout.removeWidget(self.combo)
        layout.removeWidget(self.my_label)
        layout.removeWidget(self.number)

if __name__ == "__main__":
    app = QApplication([])
    window = Window()
    window.show()
    sys.exit(app.exec())