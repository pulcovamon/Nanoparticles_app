import sys
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QGridLayout,
    QWidget, QLineEdit, QFileDialog, QMainWindow,
    QComboBox, QToolBar, QDialog, QStackedWidget,
    QFormLayout, QHBoxLayout, QVBoxLayout
)
from PyQt6.QtGui import (
    QPixmap, QDoubleValidator
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import (
    QIcon, QAction, QKeySequence, QShortcut
)
import img_fcn

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Nanoparticles app')
        self.setWindowIcon(QIcon('icon.png'))

        toolbar = QToolBar('Actions')
        toolbar.setIconSize(QSize(25, 25))
        self.addToolBar(toolbar)

        self.home = QAction(QIcon('home.png'), 'Home', self)
        self.home.triggered.connect(self.home_window)
        toolbar.addAction(self.home)

        self.images_res = QAction(QIcon('image_icon.png'),
                                    'Images', self)
        self.images_res.triggered.connect(self.img_window)
        self.images_res.setDisabled(True)
        toolbar.addAction(self.images_res)

        self.histogram = QAction(QIcon('hist.png'),
                                    'Histogram', self)
        self.histogram.triggered.connect(self.hist_window)
        self.histogram.setDisabled(True)
        toolbar.addAction(self.histogram)

        self.help = QAction(QIcon('help.png'),
                                        'help', self)
        self.help.triggered.connect(self.help_window)
        toolbar.addAction(self.help)


        main_layout = QWidget()
        self.setCentralWidget(main_layout)
            
        self.stack1 = QWidget()
        self.stack2 = QWidget()
        self.stack3 = QWidget()
        self.stack4 = QWidget()

        self.stack1UI()
        self.stack2UI()
        self.stack3UI()
            
        self.Stack = QStackedWidget(self)
        self.Stack.addWidget(self.stack1)
        self.Stack.addWidget(self.stack2)
        self.Stack.addWidget(self.stack3)
        self.Stack.addWidget(self.stack4)
            
        hbox = QHBoxLayout(main_layout)
        hbox.addWidget(self.Stack)
        self.Stack.setCurrentIndex(0)

        self.setGeometry(300, 50, 10,10)


    def home_window(self):
        self.Stack.setCurrentIndex(0)

    def img_window(self):
        self.Stack.setCurrentIndex(1)

    def hist_window(self):
        self.Stack.setCurrentIndex(2)

    def help_window(self):
        self.Stack.setCurrentIndex(3)

            
    def stack1UI(self):
        layout = QGridLayout()
        layout.setContentsMargins(10, 0, 10, 10)
        layout.setSpacing(20)

        title = QLabel('Segmentation of nanoparticles')
        title.setProperty('class', 'heading')
        layout.addWidget(title, 0, 0, 2, 6,
                                Qt.AlignmentFlag.AlignTop)

        self.add_img = QPushButton('Analyze')
        self.add_img.clicked.connect(self.add_img_fcn)
        self.add_img.setFixedSize(300, 40)
        layout.addWidget(self.add_img, 2, 1, 1, 4,
                            Qt.AlignmentFlag.AlignCenter)


        self.avg_size = QLabel('')
        self.avg_size.setProperty('class', '.normal')
        layout.addWidget(self.avg_size, 8, 1, 1, 7,
                            Qt.AlignmentFlag.AlignCenter)

        self.stack1.setLayout(layout)


    def stack2UI(self):
        layout = QFormLayout()

        name = QLabel('Histogram')
        name.setProperty('class', 'heading')
        name.setGeometry(10, 10, 30, 100)
        layout.addWidget(name)

        self.hist = QLabel(self)
        self.hist.setPixmap(QPixmap())
        self.hist.setFixedSize(600, 400)
        self.hist.setScaledContents(True)
        self.hist.move(100, 100)
        layout.addWidget(self.hist)
            
        self.stack2.setLayout(layout)


    def stack3UI(self):
        pass


    def stack4UI(self):
        layout = QFormLayout()
        title = QLabel('Documentation')
        title.setProperty('class', 'heading')
        layout.addWidget(title)

        text = QLabel('How to use this app.')
        text.set:property('class', 'normal')
        text.setWordWrap(True)
        layout.addWidget(text)
            
        self.stack4.setLayout(layout)


    def add_img_fcn(self):
        self.fname = QFileDialog.getOpenFileName(self,
            "Open File", "", "Images (*.jpg *.png *.bmp *.tiff)")

        if self.fname[0]:
            pixmap = QPixmap(self.fname[0])
            self.img.setPixmap(pixmap)
            self.img.setFixedSize(300, 300)
            self.img.setScaledContents(True)

            self.process.setDisabled(False)





       
class calculate_window(QDialog):
    def __init__(self, parent, labels):
        super().__init__(parent)

        self.setWindowTitle('Calculations')

        layout = QGridLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        self.setLayout(layout)

        self.labels = labels
        self.avg = None
        
        layout.addWidget(QLabel('Insert scale:'), 0, 0, 1, 2)
        self.scale = QLineEdit()
        self.scale.textEdited.connect(self.scale_fcn)
        self.scale.setValidator(QDoubleValidator(0, 1000, 0))
        self.scale.setFixedWidth(50)
        layout.addWidget(self.scale, 0, 3)
        layout.addWidget(QLabel('nm'), 0, 4)

        self.type_np = QComboBox()
        self.type_np.addItem('Nanoparticles')
        self.type_np.addItem('Nanorods')
        layout.addWidget(self.type_np, 1, 0, 1, 4,
                            Qt.AlignmentFlag.AlignCenter)

        self.shortcut = QShortcut(QKeySequence('Enter'), self)
        self.shortcut.setEnabled(False)
        self.shortcut.activated.connect(self.calculation_fcn)

        self.calculate = QPushButton('Calculate')
        self.calculate.setDisabled(True)
        self.calculate.clicked.connect(self.calculation_fcn)
        self.calculate.setFixedSize(QSize(300, 30))
        layout.addWidget(self.calculate, 2, 0, 1, 4,
                            Qt.AlignmentFlag.AlignCenter)


    def scale_fcn(self):
        if self.scale.text():
            self.calculate.setDisabled(False)
            self.shortcut.setEnabled(True)

    def calculation_fcn(self):
        type_text = str(self.type_np.currentText())
        self.avg = img_fcn.calculation(
                self.labels, self.scale.text(), type_text)

        self.close()

            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()

    with open('style.css', 'r') as css_file:
        app.setStyleSheet(css_file.read())

    window.show()

    try:
        sys.exit(app.exec())
    except:
        print('Exiting')