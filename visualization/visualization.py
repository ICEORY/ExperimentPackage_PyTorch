import os
from resultcurve import *
from graphgen import *
import shutil
from onlineboard import *
import datetime
import math
import os
import numpy as np


# specific manager for my experiments
class BoardManager(object):
    def __init__(self, env="main"):
        self.env = env
        self.procedure_text = TextBoard(txt_data="Total time: , Time left: , Procedure:[--------------------]0%",
                                        env=self.env)
        self.result_text = TextBoard(txt_data="Best Top1: , Best Top5: ,Loss: ", env=self.env)

        self.plot_board = [PlotBoard(plot_style=dict(title="Training Top1 Error",
                                                     xlabel="Epochs",
                                                     ylabel="Training Error (%)"),
                                     env=self.env),

                           PlotBoard(plot_style=dict(title="Training Top5 Error",
                                                     xlabel="Epochs",
                                                     ylabel="Training Error (%)"),
                                     env=self.env),

                           PlotBoard(plot_style=dict(title="Training Loss",
                                                     xlabel="Epochs",
                                                     ylabel="Loss"),
                                     env=self.env),

                           PlotBoard(plot_style=dict(title="Testing Top1 Error",
                                                     xlabel="Epochs",
                                                     ylabel="Testing Error (%)"),
                                     env=self.env),

                           PlotBoard(plot_style=dict(title="Testing Top5 Error",
                                                     xlabel="Epochs",
                                                     ylabel="Testing Error (%)"),
                                     env=self.env),

                           PlotBoard(plot_style=dict(title="Testing Loss",
                                                     xlabel="Epochs",
                                                     ylabel="Loss"),
                                     env=self.env)]

        # self.predict_scatter = ScatterBoard(env=self.env)
        print "|===>Create Board Manager Done!"

    # def updatescatter(self, x_data, y_data=None):
    #     self.predict_scatter.update(x_data, y_data)

    def updatetime(self, total_time, left_time):
        run_percent = 100 - left_time*100.0/total_time
        # print total_time, left_time
        procedure_bar = int(run_percent*0.2)
        bar_str = ""
        for i in range(20):
            if i <= procedure_bar:
                bar_str += ">"
            else:
                bar_str += "-"
        left_time_str = str(datetime.timedelta(seconds=left_time))
        total_time_str = str(datetime.timedelta(seconds=total_time))

        txt_data = "Total time: %s, Remaining Time: %s, Procedure:[%s]%.1f" % (total_time_str,
                                                                               left_time_str,
                                                                               bar_str,
                                                                               run_percent)
        txt_data += "%"

        # print txt_data
        self.procedure_text.update(txt_data=txt_data)

    def updateresult(self, result_data):
        txt_data = "Best Top1:%.4f, Best Top5: %.4f, Loss: %.4f" % (result_data[0], result_data[1], result_data[2])
        self.result_text.update(txt_data=txt_data)

    def updateplot(self, top1_error, top5_error, loss, mode):
        if mode == "Train":
            self.plot_board[0].update(y_data=top1_error)
            self.plot_board[1].update(y_data=top5_error)
            self.plot_board[2].update(y_data=loss)

        else:
            self.plot_board[3].update(y_data=top1_error)
            self.plot_board[4].update(y_data=top5_error)
            self.plot_board[5].update(y_data=loss)


class Visualization(object):
    def __init__(self, opt):
        if not os.path.isdir(opt.save_path):
            os.mkdir(opt.save_path)
        self.save_path = opt.save_path

        self.log_file = os.path.join(self.save_path, "log.txt")
        self.readme = os.path.join(self.save_path, "README.md")
        self.opt_file = os.path.join(self.save_path, "opt.log")
        self.weight_file = os.path.join(self.save_path, "weight_file.npy")
        self.code_path = os.path.join(self.save_path, "code/")
        # self.weight_folder = self.save_path + "weight/"
        # self.weight_fig_folder = self.save_path + "weight_fig/"
        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
        if os.path.isfile(self.readme):
            os.remove(self.readme)
        if not os.path.isdir(self.code_path):
            os.mkdir(self.code_path)
        self.copy_code(dst=self.code_path)
        """if os.path.isdir(self.weight_folder):
            shutil.rmtree(self.weight_folder, ignore_errors=True)
        os.mkdir(self.weight_folder)
        if os.path.isdir(self.weight_fig_folder):
            shutil.rmtree(self.weight_fig_folder, ignore_errors=True)
        os.mkdir(self.weight_fig_folder)"""

        self.graph = Graph()

        print "|===>Result will be saved at", self.save_path

    def copy_code(self, src="./", dst="./code/"):
        for file in os.listdir(src):
            file_split = file.split('.')
            if len(file_split) >= 2 and file_split[1] == "py":
                if not os.path.isdir(dst):
                    os.mkdir(dst)
                src_file = os.path.join(src, file)
                dst_file = os.path.join(dst, file)
                try:
                    shutil.copyfile(src=src_file, dst=dst_file)
                except:
                    print "copy file error! src: %s, dst: %s" % (src_file, dst_file)
            elif os.path.isdir(file):
                deeper_dst = os.path.join(dst, file)

                self.copy_code(src=os.path.join(src, file), dst=deeper_dst)

    def writeopt(self, opt):
        with open(self.opt_file, "w") as f:
            for k, v in opt.__dict__.items():
                f.write(str(k)+": "+str(v)+"\n")

    def writelog(self, input_data):
        txt_file = open(self.log_file, 'a+')
        txt_file.write(str(input_data) + "\n")
        txt_file.close()

    def writereadme(self, input_data):
        txt_file = open(self.readme, 'a+')
        txt_file.write(str(input_data) + "\n")
        txt_file.close()

    def drawcurves(self):
        drawer = DrawCurves(file_path=self.log_file, fig_path=self.save_path)
        drawer.draw(target="test_error")
        drawer.draw(target="train_error")

    def gennetwork(self, var):
        self.graph.draw(var=var)

    def savenetwork(self):
        self.graph.save(file_name=self.save_path+"network.svg")

    """def writeweights(self, input_data, block_id, layer_id, epoch_id):
        txt_path = self.weight_folder + "conv_weight_" + str(epoch_id) + ".log"
        txt_file = open(txt_path, 'a+')
        write_str = "%d\t%d\t%d\t" % (epoch_id, block_id, layer_id)
        for x in input_data:
            write_str += str(x) + "\t"
        txt_file.write(write_str+"\n")

    def drawhist(self):
        drawer = DrawHistogram(txt_folder=self.weight_folder, fig_folder=self.weight_fig_folder)
        drawer.draw()"""

    def saveweights(self, weights):
        np.save(self.weight_file, weights)
