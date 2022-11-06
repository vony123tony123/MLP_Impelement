# -*- coding: utf-8 -*-
import time
import traceback

import matplotlib
from PyQt5.QtCore import pyqtSignal, QThread

from Windows import Ui_MainWindow
from algorithm import algorithm
from resultPlotDraw import plot_design
# 导入程序运行必须模块
import sys
# PyQt5中使用的基本控件都在PyQt5.QtWidgets模块中
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import tkinter.filedialog
matplotlib.use('Qt5Agg')

# 設定gui的功能
class MyMainWindow(QMainWindow, Ui_MainWindow):

    step=0#用來判斷要不要創建plotpicture

    def choosefileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if filename:
            self.inputfilepath_edit.setText(filename)
        else:
            self.inputfilepath_edit.setText("")

    def get_info(self):
        self.pushButton.setEnabled(False)
        try:
            filepath = self.inputfilepath_edit.text()
            accuracy_value = float(self.accurary_edit.text())
            target_RMSE = float(self.RMSE_edit.text())
            learnrate = float(self.learnrate_edit.text())
            max_epochs = int(self.max_n_edit.text())
            model_structure = list(map(int, self.modelstructure_edit.text().split(',')))
            self.progressBar.setMaximum(max_epochs)

            #呼叫algorithm
            alg = algorithm(filepath, max_epochs, accuracy_value, target_RMSE, learnrate, model_structure, self.progressBar)
            printStr = "Train Accuracy = " + str(alg.train_accuracy) + "\n" + \
                       "Train RMSE = " + str(alg.train_RMSE) + "\n" + \
                       "Test Accuracy = " + str(alg.test_accuracy) + "\n"+ \
                       "Test RMSE = " + str(alg.test_RMSE) + "\n" + \
                       "Models weight = " + "\n"

            weights = list()
            for layer in alg.layers:
                weights.append(layer.getW())
                printStr = printStr + str(layer.getW())
                if layer is not alg.layers[-1]:
                    printStr = printStr + "\n------------\n"
            self.textBrowser.setText(printStr)

            self.step = self.step + 1
            if len(alg.train_data[0]) == 2:
                self.F_train.changeto2d()
                self.F_train.draw_plot_2d(alg.train_data, alg.train_ans, alg.train_predict_classify)

                self.F_test.changeto2d()
                self.F_test.draw_plot_2d(alg.test_data, alg.test_ans, alg.test_predict_classify)
                for boundary in alg.get_boundary():
                    # self.F_train.draw_function(alg.train_data, weights[-1][0])
                    # self.F_test.draw_function(alg.test_data, weights[-1][0])
                    self.F_train.draw_line(alg, alg.train_data, boundary)
                    self.F_test.draw_line(alg, alg.test_data, boundary)
            else:
                self.F_train.changeto3d()
                self.F_train.draw_plot_3d(alg.train_data, alg.train_ans, alg.train_predict_classify)
                self.F_train.draw_flat(alg.train_data, weights[-1][0])
                #self.F_train.draw_line(alg, alg.train_data)
                self.F_test.changeto3d()
                self.F_test.draw_plot_3d(alg.test_data, alg.test_ans, alg.test_predict_classify)
                self.F_test.draw_flat(alg.test_data, weights[-1][0])
                #self.F_test.draw_line(alg, alg.test_data)
            self.F_train.draw()
            self.F_test.draw()
            if self.step == 1:
                self.Train_result_plot_layout.addWidget(self.F_train)
                self.Test_result_plot_layout.addWidget(self.F_test)

            weights.clear()
            del alg
        except Exception:
            print(traceback.format_exc())
            pass

        self.pushButton.setEnabled(True)

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.inputfilepath_btn.clicked.connect(self.choosefileDialog)
        self.pushButton.clicked.connect(self.get_info)
        self.F_train = plot_design()
        self.F_test = plot_design()

if __name__ == "__main__":
    # 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    # 初始化
    myWin = MyMainWindow()

    # 将窗口控件显示在屏幕上
    myWin.show()
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())

