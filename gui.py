from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog
from python_speech_features.base import mfcc, delta
from hmmlearn import hmm
# import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import winsound
import sys
import time
import os


class HMMTrainer(object):

    def __init__(self, model_name='GMMHMM', n_components=3, min_covar=0.001, startprob_prior=1.0,
                 transmat_prior=1.0, weights_prior=1.0, means_prior=0.0, means_weight=0.0, covars_prior=None,
                 covars_weight=None, algorithm='viterbi', covariance_type='diag', random_state=None, n_iter=1000,
                 tol=0.01, verbose=False, params='stmcw', init_params='stmcw'):

        self.model_name = model_name
        self.n_components = n_components
        self.n_mix = n_mix_op
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
            print(self.n_mix)
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

    def get_means(self):
        return self.model.means_

    def get_weights(self):
        return self.model.weights_

    def get_covars(self):
        return self.model.covars_

    def get_startprob(self):
        return self.model.startprob_


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

        self.record_nm.setToolTip("Provide Information for File Path")
        self.record_nm.setDisabled(True)

        self.play_bt.setToolTip("Play Sound from Record File")
        self.play_bt.setIcon(QtGui.QIcon("icon/ico_play.png"))
        self.play_bt.setIconSize(QtCore.QSize(15, 15))
        self.play_bt.clicked.connect(self.record_play)

        self.feat13.setToolTip("Use MFCC with 13 Feature")
        self.feat13.clicked.connect(self.mfcc_option)
        self.feat39.setToolTip("Use MFCC with 39 Feature")
        self.feat39.clicked.connect(self.mfcc_option)

        self.mix3.setToolTip("Use GMM with 3 Mixture")
        self.mix3.clicked.connect(self.gmm_option)
        self.mix6.setToolTip("Use GMM with 6 Mixture")
        self.mix6.clicked.connect(self.gmm_option)

        self.select_folder_bt.setFixedSize(120, 60)
        self.select_folder_bt.setToolTip("Select folder to train")
        self.select_folder_bt.setIcon(QtGui.QIcon("icon/ico_browse.png"))
        self.select_folder_bt.setIconSize(QtCore.QSize(25, 25))
        self.select_folder_bt.clicked.connect(self.select_folder)

        self.select_folder_nm.setToolTip("Provide Information for file path")
        self.select_folder_nm.setDisabled(True)

        self.train_bt.setFixedSize(120, 60)
        self.train_bt.setToolTip("Train the provided sound with HMM-GMM")
        self.train_bt.setIcon(QtGui.QIcon("icon/ico_gear.png"))
        self.train_bt.setIconSize(QtCore.QSize(30, 30))
        self.train_bt.clicked.connect(self.train_hmm)

        self.del_bt.setFixedSize(120, 60)
        self.del_bt.setToolTip("Delete Database HMM-GMM")
        self.del_bt.setIcon(QtGui.QIcon("icon/ico_delete.png"))
        self.del_bt.setIconSize(QtCore.QSize(30, 30))
        self.del_bt.clicked.connect(self.del_train)

        self.speech_bt.setFixedSize(120, 60)
        self.speech_bt.setToolTip("Try To Speak After the Beep Sound")
        self.speech_bt.setIcon(QtGui.QIcon("icon/ico_mic.png"))
        self.speech_bt.setIconSize(QtCore.QSize(30, 30))
        self.speech_bt.clicked.connect(self.speech_stt)

    def record(self):
        global file_name
        try:
            QMessageBox.information(self, "Recording", "Record audio for 1 second")
            # record
            # print(sd.query_devices())
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
            QMessageBox.information(self, "Recording", "audio was successfully recorded")
        except Exception as e:
            print(e)

    def record_play(self):
        try:
            winsound.PlaySound(file_name, winsound.SND_FILENAME)
        except Exception as e:
            print(e)
            QMessageBox.warning(self, "Error Warning", "Please record first")

    def mfcc_option(self):
        global mfcc_coef
        if self.feat13.isChecked():
            mfcc_coef = 1
            print(mfcc_coef)
        if self.feat39.isChecked():
            mfcc_coef = 2
            print(mfcc_coef)

    def gmm_option(self):
        global n_mix_op
        if self.mix3.isChecked():
            n_mix_op = 3
            print(n_mix_op)
        if self.mix6.isChecked():
            n_mix_op = 6
            print(n_mix_op)

    def select_folder(self):
        global folder_path
        select_folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        folder_path = (select_folder + '/')
        self.select_folder_nm.setText(folder_path)

    def train_hmm(self):
        global mfcc_features
        try:
            QMessageBox.information(self, 'Train HMM-GMM', 'the audio will be train, please wait')
            t_start = time.process_time()
            input_folder = folder_path
            # print('this input', input_folder)
            print('this os list ', os.listdir(input_folder))

            for dirname in os.listdir(input_folder):
                # Get the name of the subfolder
                subfolder = os.path.join(input_folder, dirname)
                # print(subfolder)
                label = subfolder[subfolder.rfind('/') + 1:]
                print(label)

            hmm_models = []

            if os.path.exists('./model/hmm_gmm_model.pickle'):
                # "with" statements are very handy for opening files.
                with open('./model/hmm_gmm_model.pickle', 'rb') as pick:
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
                    mfcc_feat = mfcc(audio, sampling_freq)
                    if mfcc_coef == 1:
                        mfcc_features = mfcc_feat
                    if mfcc_coef == 2:
                        d_mfcc_feat = delta(mfcc_feat, 2)
                        dd_mfcc_feat = delta(d_mfcc_feat, 2)
                        mfcc_features = np.hstack([mfcc_feat, d_mfcc_feat, dd_mfcc_feat])
                    if len(X) == 0:
                        X = mfcc_features
                        # print('this 1st X', X)
                    else:
                        X = np.append(X, mfcc_features, axis=0)
                        # print('this append', X)
                    y_words.append(label)
                    # print('this Y: ', y_words)
                # print(X, end="")
                # print('X.shape =', X.shape)
                hmm_trainer = HMMTrainer()
                hmm_trainer.train(X)
                print('transmat prior ', hmm_trainer.transmat_prior)
                print('start porb prior ', hmm_trainer.startprob_prior)
                print('covarians prior ', hmm_trainer.covars_prior)
                print('weight prior ', hmm_trainer.weights_prior)
                print('means prior ', hmm_trainer.means_prior)
                print('trans mat', hmm_trainer.get_trans_mat())
                print('means ', hmm_trainer.get_means())
                print('weight', hmm_trainer.get_weights())
                print('covars ', hmm_trainer.get_covars())
                print('start prob ', hmm_trainer.get_startprob())
                hmm_models.append((hmm_trainer, label))
                with open('./model/hmm_gmm_model.pickle', 'wb') as pick:
                    pickle.dump(hmm_models, pick)
                # hmm_trainer = None
                # print(hmm_models)
            t_finish = time.process_time() - t_start
            print(t_finish, ' second')
            print(hmm_models)
            pick.close()
            QMessageBox.information(self, 'Train HMM-GMM', 'the audio was succesfully trained')
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error Warning", "Choose the File ")

    def del_train(self):
        try:
            msg = QMessageBox.question(self, 'Delete HMM', 'This action will delete HMM file',
                                       QMessageBox.Yes | QMessageBox.No)
            if msg == QMessageBox.Yes:
                print('yes')
                os.remove('D:/WORK/PROJECT/Software/PycharmProjects/TA/model/hmm_gmm_model.pickle')
            if msg == QMessageBox.No:
                print('no')
        except Exception as e:
            print(e)
            QMessageBox.information(self, 'Delete HMM', 'HMM file is not exist')

    def speech_stt(self):
        global mfcc_feature
        try:
            t_start = time.process_time()
            fs = 44100  # sample rate
            seconds = 1  # duration of record
            sd.wait(2)  # wait before recording duration start
            winsound.Beep(2000, 200)
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
            sd.wait(1)  # wait until recording duration finish
            wav.write("data_stt/stt.wav", fs, myrecording)  # save as wav file
            input_files = 'D:/WORK/PROJECT/Software/PycharmProjects/TA/data_stt/stt.wav'

            with open('./model/hmm_gmm_model.pickle', 'rb') as pick:
                hmm_models = pickle.load(pick)

            sampling_freq, audio = wav.read(input_files)
            # Extract MFCC features
            mfcc_feat = mfcc(audio, sampling_freq)
            if mfcc_coef == 1:
                mfcc_feature = mfcc_feat
            if mfcc_coef == 2:
                d_mfcc_feat = delta(mfcc_feat, 2)
                dd_mfcc_feat = delta(d_mfcc_feat, 2)
                mfcc_feature = np.hstack([mfcc_feat, d_mfcc_feat, dd_mfcc_feat])

            scores = []
            for item in hmm_models:
                hmm_model, label = item
                score = hmm_model.get_score(mfcc_feature)
                scores.append(score)
                # print('this is score: ', score)
                # print('this is scores: ', scores)
            index = np.array(scores).argmax()
            # print('this is index', index)
            predict = hmm_models[index][1]
            print("True : ", input_files)
            print("Predicted : ", predict)
            self.result.setText(predict)
            # os.remove('C:/Users/MFTA/PycharmProjects/TA/data_stt/stt.wav')
            t_finish = time.process_time() - t_start
            print(t_finish, ' second')
            pick.close()

        except Exception as e:
            print(e)
            QMessageBox.information(self, ' Info ', ' Try to speak loauder ')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
