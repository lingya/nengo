from __future__ import absolute_import, division

from datetime import timedelta
import os
import sys
import time

import numpy as np

from nengo.utils.compat import get_terminal_size


try:
    from IPython.html import widgets
    from IPython.display import display, Javascript
    import IPython.utils.traitlets as traitlets
    _HAS_WIDGETS = True
except ImportError:
    _HAS_WIDGETS = False


class Progress(object):
    def __init__(self, max_steps):
        self.steps = 0
        self.max_steps = max_steps
        self.start_time = self.end_time = time.time()
        self.finished = False

    @property
    def progress(self):
        return self.steps / self.max_steps

    @property
    def seconds_passed(self):
        if self.finished:
            return self.end_time - self.start_time
        else:
            return time.time() - self.start_time

    @property
    def eta(self):
        if self.progress > 0.:
            return (1. - self.progress) * self.seconds_passed / self.progress
        else:
            return -1

    def start(self):
        self.finished = False
        self.steps = 0
        self.start_time = time.time()

    def finish(self, success=True):
        if success:
            self.steps = self.max_steps
        self.end_time = time.time()
        self.finished = True

    def step(self, n=1):
        self.steps = min(self.steps + n, self.max_steps)


class ProgressBar(object):
    def __init__(self, update_interval=0.05):
        self.last_update = 0
        self.update_interval = update_interval

    def init(self):
        self._on_init()

    def _on_init(self):
        raise NotImplementedError()

    def update(self, progress):
        if progress.finished:
            self._on_finish(progress)
        elif self.last_update + self.update_interval < time.time():
            self._on_update(progress)
            self.last_update = time.time()

    def _on_update(self, progress):
        raise NotImplementedError()

    def _on_finish(self, progress):
        raise NotImplementedError()


class NoProgressBar(ProgressBar):
    def _on_init(self):
        pass

    def _on_update(self, progress):
        pass

    def _on_finish(self, progress):
        pass


class CmdProgressBar(ProgressBar):
    def _on_init(self):
        pass

    def _on_update(self, progress):
        line = "[{{}}] ETA: {eta}".format(eta=timedelta(
            seconds=np.ceil(progress.eta)))
        percent_str = " {}% ".format(int(100 * progress.progress))

        width, _ = get_terminal_size()
        progress_width = max(0, width - len(line))
        progress_str = (
            int(progress_width * progress.progress) * "#").ljust(
            progress_width)

        percent_pos = (len(progress_str) - len(percent_str)) // 2
        if percent_pos > 0:
            progress_str = (
                progress_str[:percent_pos] + percent_str +
                progress_str[percent_pos + len(percent_str):])

        sys.stdout.write('\r' + line.format(progress_str))
        sys.stdout.flush()

    def _on_finish(self, progress):
        width, _ = get_terminal_size()
        line = "Done in {}.".format(
            timedelta(seconds=np.ceil(progress.seconds_passed))).ljust(width)
        sys.stdout.write('\r' + line + os.linesep)
        sys.stdout.flush()


if _HAS_WIDGETS:
    class IPythonProgressWidget(widgets.DOMWidget):
        # pylint: disable=too-many-public-methods
        _view_name = traitlets.Unicode('NengoProgressBar', sync=True)
        progress = traitlets.Float(0., sync=True)
        text = traitlets.Unicode(u'', sync=True)

        FRONTEND = Javascript('''
        require(["widgets/js/widget", "widgets/js/manager"],
            function(widget, manager) {
          if (typeof widget.DOMWidgetView == 'undefined') {
            widget = IPython;
          }
          if (typeof manager.WidgetManager == 'undefined') {
            manager = IPython;
          }

          var NengoProgressBar = widget.DOMWidgetView.extend({
            render: function() {
              this.$el.css({width: '100%'});
              this.$el.html([
                '<div style="',
                    'width: 100%;',
                    'border: 1px solid #cfcfcf;',
                    'border-radius: 4px;',
                    'text-align: center;',
                    'position: relative;',
                    'margin-bottom: 0.5em;">',
                  '<div class="pb-text" style="',
                      'position: absolute;',
                      'width: 100%;">',
                    '0%',
                  '</div>',
                  '<div class="pb-bar" style="',
                      'background-color: #bdd2e6;',
                      'width: 0%;',
                      'transition: width 0.1s linear;">',
                    '&nbsp;',
                  '</div>',
                '</div>'].join(''));
            },

            update: function() {
              this.$el.css({width: '100%'});
              var progress = 100 * this.model.get('progress');
              var text = this.model.get('text');
              this.$el.find('div.pb-bar').width(progress.toString() + '%');
              this.$el.find('div.pb-text').text(text);
            },
          });

          manager.WidgetManager.register_widget_view(
            'NengoProgressBar', NengoProgressBar);
        });''')

        def _ipython_display_(self, **kwargs):
            display(self.FRONTEND)
            widgets.DOMWidget._ipython_display_(self, **kwargs)

    class IPython2ProgressBar(ProgressBar):
        def __init__(self, update_interval=0.1):
            super(IPython2ProgressBar, self).__init__(update_interval)
            self._widget = IPythonProgressWidget()

        def _on_init(self):
            display(self._widget)

        def _on_update(self, progress):
            self._widget.progress = progress.progress
            self._widget.text = "{progress:.0f}%, ETA: {eta}".format(
                progress=100 * progress.progress,
                eta=timedelta(seconds=np.ceil(progress.eta)))

        def _on_finish(self, progress):
            self._widget.progress = 1.
            self._widget.text = "Done in {}.".format(
                timedelta(seconds=np.ceil(progress.seconds_passed)))


class AutoProgressBar(ProgressBar):
    def __init__(self, delegate=None, min_eta=1.):
        if delegate is None:
            self.delegate = get_progressbar()
        else:
            self.delegate = delegate

        super(AutoProgressBar, self).__init__(self.delegate.update_interval)

        self.min_eta = min_eta
        self._visible = False

    def _on_init(self):
        pass

    def _on_update(self, progress):
        min_delay = progress.start_time + 0.1
        if self._visible:
            self.delegate.update(progress)
        elif progress.eta > self.min_eta and min_delay < time.time():
            self.delegate.init()
            self._visible = True
            self.delegate.update(progress)

    def _on_finish(self, progress):
        if self._visible:
            self.delegate.update(progress)


class ProgressControl(object):
    def __init__(self, progress, progress_bar):
        self.progress = progress
        self.progress_bar = progress_bar

    def start(self):
        self.progress.start()
        self.progress_bar.init()

    def step(self, n=1):
        self.progress.step(n)
        self.progress_bar.update(self.progress)

    def finish(self):
        self.progress.finish()
        self.progress_bar.update(self.progress)


def _in_ipynb():
    try:
        cfg = get_ipython().config  # pylint: disable=undefined-variable
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False


def get_progressbar():
    if _HAS_WIDGETS and _in_ipynb():
        return IPython2ProgressBar()
    else:
        return CmdProgressBar()
