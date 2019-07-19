from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog
from PyQt5 import QtGui
from PyQt5 import QtCore
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import winsound
import sys
import os


class MyWindow(QMainWindow):

    def __init__(self):
        super(MyWindow, self).__init__()
        loadUi('guiqt2.ui', self)

        self.title = "Speech To Text"
        self.top = 150
        self.left = 300
        self.width = 750
        self.height = 750
        self.iconName = "icon/ico_pyt.png"

        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setFixedSize(self.width, self.height)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.ui_comp()

    def ui_comp(self):
        self.recordTrain.setFixedSize(120, 60)
        self.recordTrain.setToolTip("Record Your Voice for 2 Second")
        self.recordTrain.setIcon(QtGui.QIcon("icon/ico_record.png"))
        self.recordTrain.setIconSize(QtCore.QSize(30, 30))
        self.recordTrain.clicked.connect(self.record_t)

        self.recordNameTrain.setToolTip("Provide Infomation for file path")
        self.recordNameTrain.setDisabled(True)

        self.playTrain_R.setToolTip("Play sound from Record File")
        self.playTrain_R.setIcon(QtGui.QIcon("icon/ico_play.png"))
        self.playTrain_R.setIconSize(QtCore.QSize(15, 15))
        self.playTrain_R.clicked.connect(self.record_t_play)

        self.browseTrain.setFixedSize(120, 60)
        self.browseTrain.setToolTip("Browse File .WAV")
        self.browseTrain.setIcon(QtGui.QIcon("icon/ico_browse.png"))
        self.browseTrain.setIconSize(QtCore.QSize(30, 30))
        self.browseTrain.clicked.connect(self.browse_t)

        self.browseNameTrain.setToolTip("Provide Infomation for file path")
        self.browseNameTrain.setDisabled(True)

        self.playTrain_F.setToolTip("Play sound from Record File")
        self.playTrain_F.setIcon(QtGui.QIcon("icon/ico_play.png"))
        self.playTrain_F.setIconSize(QtCore.QSize(15, 15))
        self.playTrain_F.clicked.connect(self.browse_t_play)

        self.playTrain_R.setToolTip("Play sound from Browse File")

        self.saveTrain.setFixedSize(120, 60)
        self.saveTrain.setToolTip("Save Voice As ")
        self.saveTrain.setIcon(QtGui.QIcon("icon/ico_save.png"))
        self.saveTrain.setIconSize(QtCore.QSize(25, 25))
        self.saveTrain.clicked.connect(self.save_t)

        self.saveNameTrain.setToolTip("Save Audio from source AS")

        self.trainTrain.setFixedSize(120, 60)
        self.trainTrain.setToolTip("Train the provided sound with HMM")
        self.trainTrain.setIcon(QtGui.QIcon("icon/ico_gear.png"))
        self.trainTrain.setIconSize(QtCore.QSize(30, 30))
        self.trainTrain.clicked.connect(self.train_t)

        self.fromRecordTrain.clicked.connect(self.radio_op)
        self.fromRecordTrain.setDisabled(True)

        self.fromBrowseTrain.clicked.connect(self.radio_op)
        self.fromBrowseTrain.setDisabled(True)

        self.saveNameTrain.setDisabled(True)

    def record_t(self):
        global file_path_train
        QMessageBox.information(self, "Recording", "Record audio for 1 second")
        winsound.Beep(2000, 200)
        # record
        # print(sd.query_devices())
        fs = 44100  # sample rate
        seconds = 1  # duration of record
        sd.wait(1)
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  # wait until recording duration finish
        # save record to file
        name = QFileDialog.getSaveFileName(self, 'save five', 'TA', 'sound file (*.wav)')
        getname = name[0]
        geta = name[1]
        print(getname)
        print(geta)
        wav.write(getname, fs, myrecording)  # save as wav file
        # file_name_a = wav.read("C:/Users/MFTA/PycharmProjects/TA/record/rec_train.wav")
        file_path_train= ("C:/Users/MFTA/PycharmProjects/TA/record/rec_train_vq.wav")
        self.recordNameTrain.setText(file_path_train)
        file_path = file_path_train
        QMessageBox.information(self, "Recording", "audio was succesfully recorded")
        if file_path != "":
            self.fromRecordTrain.setDisabled(False)
        else:
            pass

    def record_t_play(self):
        try:
            winsound.PlaySound(file_path_train, winsound.SND_FILENAME)
        except Exception as e:
            QMessageBox.warning(self, "Error Warning", "Please record first")

    def browse_t(self):
        global file_path_b_train
        file_name_b = QFileDialog.getOpenFileName(self, "OPEN WAV FILE", '~/users/MFTA/PycharmProjects', 'sound file (*.wav)')
        #file_name_b = QFileDialog.getExistingDirectory(self, "Select Directory", '~/users/MFTA/PycharmProjects')
        file_path_b_train = file_name_b[0]
        self.browseNameTrain.setText(file_path_b_train)
        file_path = file_path_b_train
        if file_path != "":
            self.fromBrowseTrain.setDisabled(False)
        else:
            pass

    def browse_t_play(self):
        try:
            winsound.PlaySound(file_path_b_train, winsound.SND_FILENAME)
        except Exception as e:
            QMessageBox.warning(self, "Error Warning", "Please select file first")

    def radio_op(self):
        self.saveNameTrain.setDisabled(False)
        global file_path
        if self.fromRecordTrain.isChecked() == True:
            file_path = file_path_train
            print(file_path)
        if self.fromBrowseTrain.isChecked() == True:
            file_path = file_path_b_train
            print(file_path)
        return file_path

    def save_t(self):
        choice = QMessageBox.question(self, 'Save As', "are you sure?", QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            getLabel = self.saveNameTrain.text()
            if getLabel == "":
                QMessageBox.warning(self, "Error Warning", "Must be filled")
            else:
                if os.path.exists("model/label.npy") == False:
                    print('a')
                else:
                    print('b')
        else:
            pass

    def train_t(self):
        QMessageBox.information(self, "Train HMM", "this may take some minute")
        try:
            (rate, sig) = wav.read(file_path) # read file audio wav
            mfcc_feat = mfcc(sig, rate)       # feature mfcc
            d_mfcc_feat = delta(mfcc_feat, 2)   # differential
            dd_mfcc_feat = delta(d_mfcc_feat, 2)    #acceleration
            mfcc_39_feat = np.hstack([mfcc_feat, d_mfcc_feat, dd_mfcc_feat]) #increase performance

            vq_train = mfcc_39_feat.T.reshape(1,-1) #reshape [x:39] matrix into 1d matrix by col

            #openfile [dictionary, language Model, Hidden Markov Model]
            if os.path.exists("model/codebook.npy") == False:
                np.save('model/codebook.npy', vq_train)
            else:
                cbtemp = np.load("model/codebook.npy")
                cb = np.vstack([cbtemp, vq_train])
                np.save('model/codebook.npy', cb)


        except Exception as e:
            QMessageBox.critical(self, "Error Warning", "Choose the File ")

if __name__ == '__main__':
    try:
        app = QtWidgets.QApplication(sys.argv)
        window = MyWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(e)