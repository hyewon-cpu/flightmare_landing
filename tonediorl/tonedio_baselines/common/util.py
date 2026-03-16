from shutil import copyfile
import datetime
import os
import ntpath
import numpy as np


class ConfigurationSaver:
    def __init__(self, log_dir, run_name=None):
        if run_name is not None:
            self._data_dir = log_dir + '/' + run_name
        else:
            self._data_dir = log_dir + '/' + \
                datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self._data_dir, exist_ok=True)

        # if save_items is not None:
        #     for save_item in save_items:
        #         base_file_name = ntpath.basename(save_item)
        #         copyfile(save_item, self._data_dir + '/' + base_file_name)

    @property
    def data_dir(self):
        return self._data_dir


def TensorboardLauncher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: "+url)
    webbrowser.open_new(url)
