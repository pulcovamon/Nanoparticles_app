import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QFileDialog,
    QPushButton, QLabel
)


class Window(QWidget):

    def __init__(self):
        super().__init__(parent=None)

        self.setWindowTitle("Widgets App")

        layout = QGridLayout()
        self.setLayout(layout)
        
        self.button = QPushButton('Open file')
        self.button.clicked.connect(self.clicker)
        layout.addWidget(self.button, 0, 0)

        self.label = QLabel('Filename: ')
        layout.addWidget(self.label, 1, 0)


    def clicker(self):
        fname = QFileDialog.getOpenFileName(self,
            "Open File", "", "Images (*.jpg *.png *.bmp *.tiff)")
        if fname:
            self.label.setText(fname[0])




app = QApplication(sys.argv)
window = Window()
window.show()

sys.exit(app.exec())