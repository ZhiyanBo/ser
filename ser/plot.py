import visdom

class Plot:
    
    def __init__(self, title, xlabel, ylabel, legends):
        self.vis = visdom.Visdom(port=8097)
        self.win = None
        self.opts = dict(
            title = title,
            xlabel = xlabel,
            ylabel = ylabel,
            height = 350,
            width = 500,
            ytickmin = 0,
            xtickmin = 0,
            xtickmax = 5,
            showlegend = True
        )
        self.legends = legends
    
    def update(self, legend, x, y):
        assert legend in self.legends, 'legend %s is not in %s' % (legend, self.legends)
        if self.win == None:
            self.opts['ytickmax'] = y
            self.win = self.vis.line(X=[x], Y=[y], name=legend, opts=self.opts)
        else:
            self.vis.line(X=[x], Y=[y], win=self.win, name=legend, update='append', opts=self.opts)