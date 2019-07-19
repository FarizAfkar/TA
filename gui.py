from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog
from python_speech_features import mfcc, delta
from hmmlearn import hmm
# import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import winsound
import sys
import os


class HMMTrainer(object):

    def __init__(self, model_name='GMMHMM', n_components=4, n_mix=5, min_covar=0.001, startprob_prior=1.0,
                 transmat_prior=1.0, weights_prior=1.0, means_prior=0.0, means_weight=0.0, covars_prior=None,
                 covars_weight=None, algorithm='viterbi', covariance_type='diag', random_state=None, n_iter=1000,
                 tol=0.01, verbose=False, params='stmcw', init_params='stmcw'):

        self.model_name = model_name
        self.n_components = n_components
        self.n_mix = n_mix
        self.min_covar = min_covar
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.weights_prior = weights_prior
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.algorithm = algorithm
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.params = params
        self.init_params = init_params
        self.models = []

        if self.model_name == 'GMMHMM':
            self.model = hmm.GMMHMM(n_components=self.n_components, n_mix=self.n_mix,
                                    covariance_type=self.covariance_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)

    # get transition matrix
    def get_trans_mat(self):
        return self.model.transmat_

    # get probability matrix
    def get_prob(self, input_data):
        return self.model.predict_proba(input_data)


class MyWindow(QMainWindow):

    def __init__(self):
        super(MyWindow, self).__init__()
        loadUi('guiqtfinal.ui', self)

        self.title = "Speech To Text"
        self.top = 150
        self.left = 300
        self.width = 1400
        self.height = 650
        self.iconName = "icon/ico_stt.png"

        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setFixedSize(self.width, self.height)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.ui_comp()

    def ui_comp(self):
        self.record_bt.setFixedSize(120, 60)
        self.record_bt.setToolTip("Record Your Voice for 1 Second")
        self.record_bt.setIcon(QtGui.QIcon("icon/ico_record.png"))
        self.record_bt.setIconSize(QtCore.QSize(30, 30))
        self.record_bt.clicked.connect(self.record)

        self.record_nm.setToolTip("Provide Infomation for file path")
        self.record_nm.setDisabled(True)

        self.play_bt.setToolTip("Play sound from Record File")
        self.play_bt.setIcon(QtGui.QIcon("icon/ico_play.png"))
        self.play_bt.setIconSize(QtCore.QSize(15, 15))
        self.play_bt.clicked.connect(self.record_play)

        self.select_folder_bt.setFixedSize(120, 60)
        self.select_folder_bt.setToolTip("Select folder to train")
        self.select_folder_bt.setIcon(QtGui.QIcon("icon/ico_browse.png"))
        self.select_folder_bt.setIconSize(QtCore.QSize(25, 25))
        self.select_folder_bt.clicked.connect(self.select_folder)

        self.select_folder_nm.setToolTip("Provide Infomation for file path")
        self.select_folder_nm.setDisabled(True)

        self.train_bt.setFixedSize(120, 60)
        self.train_bt.setToolTip("Train the provided sound with HMM")
        self.train_bt.setIcon(QtGui.QIcon("icon/ico_gear.png"))
        self.train_bt.setIconSize(QtCore.QSize(30, 30))
        self.train_bt.clicked.connect(self.train_hmm)

        self.speech_bt.setFixedSize(120, 60)
        self.speech_bt.setToolTip("Speech to Text")
        self.speech_bt.setIcon(QtGui.QIcon("icon/ico_mic.png"))
        self.speech_bt.setIconSize(QtCore.QSize(30, 30))
        self.speech_bt.clicked.connect(self.speech_stt)

    def record(self):
        global file_name
        try:
            QMessageBox.information(self, "Recording", "Record audio for 1 second")
            # record
            # print(sd.query_devices()) << for detect sound driver
            fs = 44100  # sample rate
            seconds = 1  # duration of record
            sd.wait(2)  # wait before recording duration start
            winsound.Beep(2000, 200)
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
            sd.wait(1)  # wait until recording duration finish
            # save record to file
            get_name = QFileDialog.getSaveFileName(self, 'save five', 'TA', 'sound file (*.wav)')
            file_name = get_name[0]
            wav.write(file_name, fs, myrecording)  # save as wav file
            self.record_nm.setText(file_name)
            QMessageBox.information(self, "Recording", "audio was succesfully recorded")
        except Exception as e:
            print(e)

    def record_play(self):
        try:
            winsound.PlaySound(file_name, winsound.SND_FILENAME)
        except Exception as e:
            print(e)
            QMessageBox.warning(self, "Error Warning", "Please record first")

    def select_folder(self):
        global folder_path
        select_folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        folder_path = (select_folder + '/')
        self.select_folder_nm.setText(folder_path)

    def train_hmm(self):
        try:
            QMessageBox.information(self, 'Train HMM', 'the audio will be train, please wait')
            input_folder = folder_path
            print('this input', input_folder)
            print('this os list ', os.listdir(input_folder))

            for dirname in os.listdir(input_folder):
                # Get the name of the subfolder
                subfolder = os.path.join(input_folder, dirname)
                # print(subfolder)
                label = subfolder[subfolder.rfind('/') + 1:]
                print(label)

            hmm_models = []

            if os.path.exists('./model/hmm_model.pickle'):
                # "with" statements are very handy for opening files.
                with open('./model/hmm_model.pickle', 'rb') as pick:
                    hmm_models = pickle.load(pick)

            for dirname in os.listdir(input_folder):
                subfolder = os.path.join(input_folder, dirname)
                if not os.path.isdir(subfolder):
                    continue
                label = subfolder[subfolder.rfind('/') + 1:]
                X = np.array([])
                y_words = []
                for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
                    filepath = os.path.join(subfolder, filename)
                    sampling_freq, audio = wav.read(filepath)
                    mfcc_features = mfcc(audio, sampling_freq)
                    d_mfcc_feat = delta(mfcc_features, 2)
                    dd_mfcc_feat = delta(d_mfcc_feat, 2)
                    mfcc_39 = np.hstack([mfcc_features, d_mfcc_feat, dd_mfcc_feat])
                    if len(X) == 0:
                        X = mfcc_39
                        # print('this 1st X', X)
                    else:
                        X = np.append(X, mfcc_39, axis=0)
                        # print('this append', X)
                    y_words.append(label)
                    # print('this Y: ', y_words)
                # print(X, end="")
                # print('X.shape =', X.shape)
                hmm_trainer = HMMTrainer()
                hmm_trainer.train(X)
                hmm_models.append((hmm_trainer, label))
                with open('./model/hmm_model.pickle', 'wb') as pick:
                    pickle.dump(hmm_models, pick)
                # hmm_trainer = None
                # print(hmm_models)
            pick.close()
            QMessageBox.information(self, 'Train Hmm', 'the audio was succesfully trained')
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error Warning", "Choose the File ")

    def speech_stt(self):
        try:
            fs = 44100  # sample rate
            seconds = 1  # duration of record
            winsound.Beep(1500, 300)
            sd.wait(5)  # wait before recording duration start
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
            wav.write("speech/speech.wav", fs, myrecording)  # save as wav file
            input_files = ['C:/Users/MFTA/PycharmProjects/TA/speech/speech.wav']
            with open('./model/hmm_model.pickle', 'rb') as pick:
                hmm_models = pickle.load(pick)

            for input_file in input_files:
                sampling_freq, audio = wav.read(input_file)
                # Extract MFCC features
                mfcc_features = mfcc(audio, sampling_freq)
                d_mfcc_feat = delta(mfcc_features, 2)
                dd_mfcc_feat = delta(d_mfcc_feat, 2)
                mfcc_39 = np.hstack([mfcc_features, d_mfcc_feat, dd_mfcc_feat])

                scores = []
                for item in hmm_models:
                    hmm_model, label = item
                    score = hmm_model.get_score(mfcc_39)
                    scores.append(score)
                    # print('this is score: ', score)
                    # print('this is scores: ', scores)
                index = np.array(scores).argmax()
                # print('this is index', index)
                predict = hmm_models[index][1]
                print("True : ", input_file)
                print("Predicted : ", predict)
                self.result.setText(predict)
            os.remove('C:/Users/MFTA/PycharmProjects/TA/speech/speech.wav')
            pick.close()
        except Exception as e:
            print(e)
            QMessageBox.information(self, ' Info ', ' Try to speak loauder ')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
