# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use('Qt5Agg')

#創建一個matplotlib圖形繪製類
class plot_design(FigureCanvas):
    def __init__(self,width=9,height=8,dpi=100):
        # 第一步：創建一個創建Figure
        self.fig=Figure(figsize=(width,height),dpi=dpi)
        # 第二步：在父類中激活Figure窗口
        super(plot_design,self).__init__(self.fig)#此句必不可少，否則不能顯示圖形
        # 第三步：創建一個子圖，用於繪製圖形用
        # 111表示在1*1網格中第1個子圖
        self.axes_answer = self.fig.add_subplot(121)
        self.axes_predict = self.fig.add_subplot(122)
        # 第四步：就是畫圖，可以在此類中畫，也可以在其它類中畫,最好是在別的地方作圖
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    #畫data的plot，第一群為藍色，第二群為紅色，以外的為綠色
    def draw_plot_2d(self, data_list, ans_list, predict_list):
        self.axes_answer.clear()
        self.axes_predict.clear()
        self.axes_answer.grid()
        self.axes_predict.grid()

        max_x=np.amax(data_list,axis=0)
        min_x=np.amin(data_list,axis=0)

        np_data_list = np.array(data_list)
        self.axes_answer.scatter(np_data_list[:,0],np_data_list[:,1],c=ans_list, alpha=1)
        self.axes_answer.set_title('Answer')
        self.axes_answer.set_xlim(min_x[0]-0.5, max_x[0]+0.5)
        self.axes_answer.set_ylim(min_x[1]-0.5, max_x[1]+0.5)
        self.axes_predict.scatter(np_data_list[:,0],np_data_list[:,1],c=predict_list, alpha=1)
        self.axes_predict.set_title('Predict')
        self.axes_predict.set_xlim(min_x[0]-0.5, max_x[0]+0.5)
        self.axes_predict.set_ylim(min_x[1]-0.5, max_x[1]+0.5)
        self.draw()

        if len(np_data_list[:,1]) > 1:
            self.axes_predict.autoscale(enable=False, axis='both', tight=False)  # 讓畫布大小固定
            self.axes_answer.autoscale(enable=False, axis='both', tight=False)

    def draw_plot_3d(self, data_list, ans_list, predict_list):
        self.axes_answer.clear()
        self.axes_predict.clear()
        self.axes_answer.grid()
        self.axes_predict.grid()

        max_x = np.amax(data_list, axis=0)
        min_x = np.amin(data_list, axis=0)

        np_data_list = np.array(data_list)
        self.axes_answer.scatter(np_data_list[:, 0], np_data_list[:, 1], np_data_list[:, 2], c=ans_list, alpha=1)
        self.axes_answer.set_title('Answer')
        self.axes_answer.set_xlim(min_x[0] - 0.5, max_x[0] + 0.5)
        self.axes_answer.set_ylim(min_x[1] - 0.5, max_x[1] + 0.5)
        self.axes_predict.scatter(np_data_list[:, 0], np_data_list[:, 1], np_data_list[:,2],c=predict_list, alpha=1)
        self.axes_predict.set_title('Predict')
        self.axes_predict.set_xlim(min_x[0] - 0.5, max_x[0] + 0.5)
        self.axes_predict.set_ylim(min_x[1] - 0.5, max_x[1] + 0.5)
        self.draw()

        if len(np_data_list[:, 1]) > 1:
            self.axes_predict.autoscale(enable=False, axis='both', tight=False)  # 讓畫布大小固定
            self.axes_answer.autoscale(enable=False, axis='both', tight=False)

    def draw_function(self, data_list, output_w):
        w0=output_w[0]
        w1=output_w[1]
        w2=output_w[2]

        #axis=軸，axis0為row，axis1為col
        max=np.amax(data_list,axis=0)
        min=np.amin(data_list,axis=0)
        if w2!=0:
            x1=np.linspace(min[0]-2,max[0]+2,num=100)
            x2=(w0-w1*x1)/w2
            self.axes_answer.plot(x1,x2)
            self.axes_predict.plot(x1,x2)
        elif w1!=0:
            x2=np.linspace(min[1]-2,max[1]+2,num=100)
            x1=(w0-w2*x2)/w1
            self.axes_answer.plot(x1,x2)
            self.axes_predict.plot(x1,x2)
        else:
            print('Error,No fuction')
        self.draw()

    def draw_flat(self, data_list, output_w):
        w0=output_w[0]
        w1=output_w[1]
        w2=output_w[2]
        w3=output_w[3]
        #axis=軸，axis0為row，axis1為col
        max=np.amax(data_list,axis=0)
        min=np.amin(data_list,axis=0)

        x1 = np.linspace(min[0]-0.5,max[0]+0.5,num=100)
        x2 = np.linspace(min[1]-0.5,max[1]+0.5,num=100)
        X1, X2 = np.meshgrid(x1, x2)
        X3 = (w0 - w1 * X1 - w2 * X2)/w3

        self.axes_answer.plot_surface(X1,X2,X3, color=(1, 0, 0, 0.5))
        self.axes_predict.plot_surface(X1,X2,X3, color=(1, 0, 0, 0.5))
        self.draw()


    def draw_line(self, alg, input_data, boundary):
        # axis=軸，axis0為row，axis1為col
        max = np.amax(input_data, axis=0)
        min = np.amin(input_data, axis=0)
        x_all = np.linspace(min[0]-1,max[0]+1,num=100)
        y_all = np.linspace(min[1]-1,max[1]+1,num=100)
        dataset = list()
        for x in x_all:
            for y in y_all:
                dataset.append([x,y])
        predict_value, predict_classify = alg.predict(dataset)
        plotpoints = np.array([data for data, pred in zip(dataset, predict_value) if round(pred,1)== round(boundary,1)])
        # self.axes_answer.plot(plotpoints[:,0],plotpoints[:,1])
        # self.axes_predict.plot(plotpoints[:,0],plotpoints[:,1])
        if plotpoints is []:
            return
        linepoints = list()
        for x1 in list(set(plotpoints[:,0])):
            samex1points = np.array([point[1] for point in plotpoints if point[0]==x1])
            midx2 = np.median(samex1points)
            linepoints.append((x1, midx2))
        linepoints.sort(key=lambda a: a[0])
        linepoints = np.array(linepoints)
        self.axes_answer.plot(linepoints[:,0],linepoints[:,1])
        self.axes_predict.plot(linepoints[:,0],linepoints[:,1])
        self.draw()


    def changeto3d(self):
        self.fig.delaxes(self.axes_answer)
        self.fig.delaxes(self.axes_predict)
        self.axes_answer = self.fig.add_subplot(121, projection='3d')
        self.axes_predict = self.fig.add_subplot(122, projection='3d')

    def changeto2d(self):
        self.fig.delaxes(self.axes_answer)
        self.fig.delaxes(self.axes_predict)
        self.axes_answer = self.fig.add_subplot(121)
        self.axes_predict = self.fig.add_subplot(122)