# -*- coding:utf-8 -*-
import copy
import math
import random

import numpy as np
from PyQt5.QtWidgets import QApplication
from sklearn.model_selection import train_test_split


class algorithm():
    label_list = list()
    function = list()
    train_accuracy = 0
    test_accuracy = 0

    # 計算方程式的係數
    def funtion_value(self):
        self.function = copy.deepcopy(self.layers[-1].getW())
        self.function[0] *= -1

    def __init__(self, data_path, max_epochs, accuracy, RMSE, learnrate, model_structure, progressBar):
        max_epochs = max_epochs # 最大運轉次數
        target_accuracy = accuracy # 目標準確度
        target_RMSE = RMSE
        self.target_data = list()
        self.target_answer = list()
        self.target_data, self.target_answer = self.open_file(data_path)
        self.label_list, self.target_answer = self.standard(self.target_answer)

        self.train_data, self.test_data, self.train_ans, self.test_ans = train_test_split(self.target_data,
                                                                                          self.target_answer,
                                                                                          test_size=0.3)
        self.train_predict_classify = list()
        self.train_predict_value = list()
        self.test_predict_classify = list()
        self.test_predict_value = list()

        self.training_data(self.train_data, self.train_ans, learnrate, max_epochs, target_accuracy, target_RMSE, model_structure, progressBar)
        self.testing_data(self.test_data, self.test_ans)
        print(self.train_accuracy)
        print(self.test_accuracy)

    # 進行training,先調整權重，後計算準確度
    def training_data(self, train_datas, train_ans, learnrate, epochs, target_accuracy, target_RMSE, model_structure, progressBar):
        self.layers = list()
        input_dims = len(train_datas[0])
        for i in range(len(model_structure)):
            self.layers.append(Layer(model_structure[i], input_dims))
            input_dims = model_structure[i]

        for epoch in range(epochs):
            print("Epoch : " + str(epoch) +'/'+str(epochs))
            QApplication.processEvents()
            progressBar.setValue(epoch)
            for input_data, ans in zip(train_datas, train_ans):
                y = input_data
                for layer in self.layers:
                    y = layer.feedForward(y)
                gradient =self.layers[-1].output_gradient(ans, y)
                w = self.layers[-1].getW()
                self.layers[-1].optimizer(learnrate)
                for layer in self.layers[-2:]:
                    gradient = layer.hidden_gradient(gradient, np.array(w)[:, 1:])
                    w = layer.getW()
                    layer.optimizer(learnrate)

            # 計算這次epoch的Accuracy
            self.train_predict_value.clear()
            self.train_predict_classify.clear()
            self.train_RMSE, self.train_accuracy, self.train_predict_value, self.train_predict_classify = self.evaluate(train_datas, train_ans)
            if self.train_RMSE <= target_RMSE:
                print("early end the RMSE is " + str(self.train_RMSE))
                break
            if self.train_accuracy >= target_accuracy:
                print("early end the training accuracy is " + str(self.train_accuracy))
                break
        progressBar.setValue(epochs)

    # 計算testing_data，並記錄準確度
    def testing_data(self, test_data, test_ans):
        self.test_RMSE, self.test_accuracy, self.test_predict_value, self.test_predict_classify = self.evaluate(test_data, test_ans)

    def open_file(self, datapath):
        datas = list()
        answers = list()
        with open(datapath, 'r') as fout:
            for lines in fout.readlines():
                lines = list(map(float, lines.strip('\n').split(' ')))
                datas.append(lines[0:-1])
                answers.append(int(lines[-1]))
        return datas, answers

    def standard(self, answers):
        original_label_list = list(set(answers))
        num_labels = len(original_label_list)
        standard_labels = list([i / (num_labels - 1) for i in range(num_labels)])
        standard_answer = list([standard_labels[original_label_list.index(ans)] for ans in answers])
        return standard_labels, standard_answer

    def classify(self, y):
        classify_bounds = list([i / len(self.label_list) for i in range(len(self.label_list) + 1)])
        for i in range(len(classify_bounds)):
            if classify_bounds[i] >= y:
                pred = self.label_list[i - 1]
                return pred

    def evaluate(self, input_datas, input_ans):
        correct = 0
        MSE = 0
        input_predict_value = list()
        input_predict_classify = list()
        for input_data, ans in zip(input_datas, input_ans):
            y = input_data
            for layer in self.layers:
                y = layer.feedForward(y)
            y = y[0]
            input_predict_value.append(y)
            pred = self.classify(y)
            input_predict_classify.append(pred)
            if pred == ans:
                correct += 1
            MSE += math.pow((y - ans), 2) / len(input_datas)
        RMSE = math.pow(MSE, 0.5)
        accuracy = correct / len(input_datas)
        return RMSE, accuracy, input_predict_value, input_predict_classify

    def predict(self, input_datas):
        input_predict_value = list()
        input_predict_classify = list()
        for input_data in input_datas:
            y = input_data
            for layer in self.layers:
                y = layer.feedForward(y)
            y = y[0]
            input_predict_value.append(y)
            pred = self.classify(y)
            input_predict_classify.append(pred)
        return input_predict_value, input_predict_classify

    def get_boundary(self):
        classify_bounds = list([i / len(self.label_list) for i in range(len(self.label_list) + 1)])
        return classify_bounds[1:-1]

    # 當刪除object時，list歸零
    def __del__(self):
        self.train_data.clear()
        self.test_data.clear()
        self.train_ans.clear()
        self.test_ans.clear()
        self.label_list.clear()
        self.layers.clear()

class Layer:
    def __init__(self, num_nodes, input_dim):
        self.gradient = None
        weight = list()
        for x in range(num_nodes):
            w = list([np.random.rand() for x in range(input_dim + 1)])
            weight.append(w)
        self.weight = weight

    def feedForward(self, input_data):
        self.x = np.insert(input_data, 0, -1)
        self.y = list(map(self.sigmod, np.dot(self.weight, self.x)))
        return self.y

    def sigmod(self, v):
        return 1 / (1 + math.exp(-1 * v))

    def sigmod_dot(self, y):
        return y * (1 - y)

    def output_gradient(self, ans, pred):
        self.gradient = list(map(lambda d, y:(d - y) * self.sigmod_dot(y), list([ans]), pred))
        return self.gradient

    def hidden_gradient(self, last_gradient, w):
        sigmod_dot = list(map(self.sigmod_dot, self.y))
        self.gradient = list(map(lambda x, y: x * y, sigmod_dot, np.dot(last_gradient, w)))
        return self.gradient

    def optimizer(self, learnrate):
        self.weight = list(map(lambda weight, gradient: weight + learnrate * gradient * self.x, self.weight, self.gradient))
        return self.weight

    def getW(self):
        return self.weight
