import sys
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QGridLayout,
    QWidget, QLineEdit, QFileDialog, QMainWindow,
    QComboBox, QToolBar, QDialog, QStackedWidget,
    QFormLayout, QHBoxLayout, QVBoxLayout,
    QSpinBox
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
        self.setWindowIcon(QIcon('icons/icon.png'))

        toolbar = QToolBar('Actions')
        toolbar.setIconSize(QSize(25, 25))
        self.addToolBar(toolbar)

        self.home = QAction(QIcon('icons/home.png'), 'Home', self)
        self.home.triggered.connect(self.home_window)
        toolbar.addAction(self.home)

        self.l_image = QAction(QIcon('icons/image_icon.png'), 'Labeled image', self)
        self.l_image.triggered.connect(self.labeled_window)
        self.l_image.setDisabled(True)
        toolbar.addAction(self.l_image)

        self.histogram = QAction(QIcon('icons/hist.png'),
                                    'Histogram', self)
        self.histogram.triggered.connect(self.hist_window)
        self.histogram.setDisabled(True)
        toolbar.addAction(self.histogram)

        self.help = QAction(QIcon('icons/help.png'),
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
        self.stack4UI()
            
        self.Stack = QStackedWidget (self)
        self.Stack.addWidget (self.stack1)
        self.Stack.addWidget (self.stack2)
        self.Stack.addWidget (self.stack3)
        self.Stack.addWidget (self.stack4)
            
        hbox = QHBoxLayout(main_layout)
        hbox.addWidget(self.Stack)
        self.Stack.setCurrentIndex(0)

        self.setGeometry(300, 50, 10,10)


    def home_window(self):
        self.Stack.setCurrentIndex(0)

    def hist_window(self):
        self.Stack.setCurrentIndex(1)

    def labeled_window(self):
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

        self.add_img = QPushButton('Upload images')
        self.add_img.clicked.connect(self.add_img_fcn)
        self.add_img.setFixedSize(400, 40)
        layout.addWidget(self.add_img, 2, 0, 1, 7,
                            Qt.AlignmentFlag.AlignCenter)

        self.process = QPushButton('Analyze images')
        self.process.clicked.connect(self.processing)
        self.process.setDisabled(True)
        self.process.setFixedSize(400, 40)
        layout.addWidget(self.process, 2, 7, 1, 7,
                            Qt.AlignmentFlag.AlignCenter)

        self.img = QLabel(self)
        self.img.setPixmap(QPixmap())
        self.img.setFixedSize(400, 400)
        self.img.setScaledContents(True)
        layout.addWidget(self.img, 3, 0, 7, 7,
                            Qt.AlignmentFlag.AlignCenter)

        type_text = QLabel('Particle type:')
        layout.addWidget(type_text, 4, 8, 1, 2)

        self.type_w = QComboBox()
        self.type_w.addItem('nanoparticles')
        self.type_w.addItem('nanorods')
        layout.addWidget(self.type_w, 4, 10, 1, 3,
                            Qt.AlignmentFlag.AlignCenter)

        scale_text = QLabel('Image scale:')
        layout.addWidget(scale_text, 6, 8, 1, 2)

        self.scale_w = QSpinBox()
        self.scale_w.setValue(100)
        self.scale_w.setMinimum(10)
        self.scale_w.setMaximum(1000)
        self.scale_w.setSingleStep(10)
        layout.addWidget(self.scale_w, 6, 10, 1, 2,
                            Qt.AlignmentFlag.AlignCenter)

        nm = QLabel('nm')
        layout.addWidget(nm, 6, 12, 1, 1)

        self.avg_size = QLabel('')
        self.avg_size.setProperty('class', '.normal')
        layout.addWidget(self.avg_size, 11, 1, 1, 12,
                            Qt.AlignmentFlag.AlignCenter)

        self.avg_size2 = QLabel('')
        self.avg_size2.setProperty('class', '.normal')
        layout.addWidget(self.avg_size2, 10, 1, 1, 12,
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
        self.hist.setFixedSize(800, 500)
        self.hist.setScaledContents(True)
        self.hist.move(100, 100)
        layout.addWidget(self.hist)
            
        self.stack2.setLayout(layout)


    def stack4UI(self):
        layout = QFormLayout()
        title = QLabel('About this app')
        title.setProperty('class', 'heading')
        layout.addWidget(title)

        text = QLabel('About this app.')
        text.set:property('class', 'normal')
        text.setWordWrap(True)
        layout.addWidget(text)
            
        self.stack4.setLayout(layout)


    def stack3UI(self):
        layout = QFormLayout()
        title = QLabel('Labeled image')
        title.setProperty('class', 'heading')
        layout.addWidget(title)

        self.labeled = QLabel(self)
        self.labeled.setPixmap(QPixmap())
        self.labeled.setFixedSize(600, 600)
        self.labeled.setScaledContents(True)
        layout.addWidget(self.labeled)

        self.stack3.setLayout(layout)


    def add_img_fcn(self):
        self.fname = QFileDialog.getOpenFileNamesAndFilter(self,
            "Open Files", "", "Images (*.jpg *.png *.bmp *.tiff)")

        if self.fname[0]:
            pixmap = QPixmap(self.fname[0])
            self.img.setPixmap(pixmap)
            self.img.setFixedSize(400, 400)
            self.img.setScaledContents(True)

            self.process.setDisabled(False)



    def processing(self):
        
        np_type = self.type_w.currentText()
        scale = self.scale_w.value()
        input_desc = {'image': [scale, np_type, self.fname[0]]}

        labeledf, histf, sizesf = img_fcn.image_analysis(input_desc, 'image')

        pixmap = QPixmap(labeledf)
        self.labeled.setPixmap(pixmap)
        self.labeled.setFixedSize(600, 600)
        self.labeled.setScaledContents(True)

        pixmap = QPixmap(histf)
        self.hist.setPixmap(pixmap)
        self.hist.setFixedSize(800, 500)
        self.hist.setScaledContents(True)

        self.histogram.setDisabled(False)
        self.l_image.setDisabled(False)

        with open(sizesf, mode='r') as txt_file:
            sizes = txt_file.readlines()

        if np_type == 'nanoparticles':
            self.avg_size.setText(sizes[0])
        elif np_type == 'nanorods':
            self.avg_size.setText(sizes[2])
            self.avg_size2.setText(sizes[1])
          

            
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