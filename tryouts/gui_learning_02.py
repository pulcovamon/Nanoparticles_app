import sys
from PyQt6.QtWidgets import ( 
    QApplication, QWidget, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout
)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon



class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GUI learning 02')
        self.setWindowIcon(QIcon('icon.png'))
        self.setContentsMargins(20, 20, 20, 20)

        layout = QGridLayout()
        self.setLayout(layout)

        button1 = QPushButton('Click me', self)             # signals (sent by button) and slots (trigger activates a slot - function connected to the button)
        button1.clicked.connect(self.close)
        button2 = QPushButton('Do not click me', self)
        button2.clicked.connect(self.hello)
        button3 = QPushButton('Also me', self)
        button3.released.connect(self.hello)
        button4 = QPushButton('Do not dare!', self)
        button4.setDisabled(True)
        button4.pressed.connect(self.hello)
        layout.addWidget(button1, 0, 0)
        layout.addWidget(button2, 0, 1, 1, 2)
        layout.addWidget(button3, 1, 0)
        layout.addWidget(button4, 1, 1, 1, 2)


    def hello(self):
        print('Hello world')



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())