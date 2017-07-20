import visdom
import numpy as np
import time


class OnlineBoard(object):
    def __init__(self, env):
        try:
            self.viz = visdom.Visdom()
        except:
            self.viz = None
            print "Enter Cmd: python -m visdom.server on shell"
        self.env = env
        self.win = None


class TextBoard(OnlineBoard):
    def __init__(self, txt_data="Text board initializing ...", env="main"):
        super(TextBoard, self).__init__(env)
        self.create(txt_data=txt_data)

    def create(self, txt_data):
        self.win = self.viz.text(text=txt_data, env=self.env)

    def update(self, txt_data):
        self.viz.text(text=txt_data, win=self.win, env=self.env)


class PlotBoard(OnlineBoard):
    def __init__(self, env="main", plot_style=None):
        super(PlotBoard, self).__init__(env)
        self.plot_style = plot_style or dict(title="Online Board", xlabel="Epochs", ylabel="Error")
        self.x_data = 1

    def update(self, y_data, x_data=None):

        if not isinstance(y_data, np.ndarray):
            if isinstance(y_data, list):
                y_data = np.array(y_data)
            else:
                y_data = np.array([y_data])
            y_data = y_data.transpose()

        # print y_data.shape

        if x_data is None:
            x_data_cell = np.array([range(self.x_data, self.x_data+len(y_data))])
            x_data_cell = x_data_cell.transpose()
            x_data = x_data_cell
            if len(y_data.shape) > 1:
                for i in range(1, y_data.shape[1]):
                    x_data = np.column_stack((x_data, x_data_cell))
            else:
                x_data = x_data.squeeze(1)
        # print x_data.shape

        if self.win is None:
            self.win = self.viz.line(Y=y_data, X=x_data, env=self.env, opts=self.plot_style)
        else:
            self.viz.updateTrace(X=x_data, Y=y_data, env=self.env, win=self.win)

        self.x_data += len(y_data)


class SVGBoard(OnlineBoard):
    def __init__(self, svg_file, env="main"):
        super(SVGBoard, self).__init__(env)
        self.create(svg_file)

    def create(self, svg_file):
        self.win = self.viz.svg(svgfile=svg_file, env=self.env)

    def update(self, svg_file):
        self.viz.svg(svgfile=svg_file, env=self.env, win=self.win)


class ScatterBoard(OnlineBoard):
    def __init__(self, env="main", opts=None):
        super(ScatterBoard, self).__init__(env)
        self.opts = opts or dict(markersize=5)

    def create(self, x_data, y_data=None):
        self.win = self.viz.scatter(x_data, y_data, opts=self.opts, env=self.env)

    def update(self, x_data, y_data=None):
        if self.win is None:
            self.create(x_data, y_data)
        else:
            self.viz.scatter(x_data, y_data, win=self.win, env=self.env, opts=self.opts)
