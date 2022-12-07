import sys
from PyQt6.QtWidgets import ( 
    QApplication, QWidget, QPushButton, QLabel,
    QLineEdit, QGridLayout
)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
import csv


class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('GUI learning')
        self.setWindowIcon(QIcon('icon.png'))

        layout = QGridLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        self.setLayout(layout)

        title = QLabel('Login Form with PyQt6')
        title.setProperty('class', 'heading')
        layout.addWidget(title, 0, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)

        user = QLabel('Username:')
        user.setProperty('class', 'normal')
        layout.addWidget(user, 1, 0)

        pwd = QLabel('Password:')
        pwd.setProperty('class', 'normal')
        layout.addWidget(pwd, 2, 0)

        self.input1 = QLineEdit()
        self.input1.textEdited.connect(self.login_butt)
        layout.addWidget(self.input1, 1, 1, 1, 2)

        self.input2 = QLineEdit()
        self.input2.setEchoMode(QLineEdit.EchoMode.Password)
        self.input2.textEdited.connect(self.login_butt)
        layout.addWidget(self.input2, 2, 1, 1, 2)

        self.button1 = QPushButton('Register')
        self.button1.clicked.connect(self.register)
        self.button1.setDisabled(True)
        layout.addWidget(self.button1, 3, 1)

        self.button2 = QPushButton('Login')
        self.button2.clicked.connect(self.login)
        self.button2.setDisabled(True)
        layout.addWidget(self.button2, 3, 2)

        self.status = QLabel('')
        self.status.setProperty('class', 'stat')
        layout.addWidget(self.status, 4, 0, 1, 3, alignment=Qt.AlignmentFlag.AlignCenter)


    def register(self):
        header = ['ID', 'USERNAME', 'PASSWORD', 'EMAIL']
        user = {'USERNAME': self.input1.text(), 'PASSWORD': self.input2.text()}

        with open('users.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, delimiter=';', fieldnames=header)
            csv_writer.writerow(user)

        self.status.setText('Registration successful.')



    def login(self):
        with open('users.csv', 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=';')
            user = False

            for line in csv_reader:
                if line['name'] == self.input1.text():
                    if line['password'] == self.input2.text():
                        print('successful')
                        self.close()
                        user = True
                    break

            if not user:
                print('unknown account or wrong password')
                self.status.setText('Wrong username or password.')


    def login_butt(self):

        if self.input1.text() and self.input2.text():
            self.button2.setDisabled(False)
            self.button1.setDisabled(False)
        else:
            self.button2.setDisabled(True)
            self.button1.setDisabled(True)




app = QApplication(sys.argv)
window = Window()

with open('style.css', 'r') as css_file:
    app.setStyleSheet(css_file.read())

window.show()
sys.exit(app.exec())