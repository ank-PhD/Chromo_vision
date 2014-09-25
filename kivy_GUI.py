from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.progressbar import ProgressBar
from kivy.utils import platform
from configs import loop_dir, loop_fle, afterloop

import os, errno

class MyWidget(BoxLayout):
    progress_bar = ProgressBar(max=1000)

    def __init__(self, **kwargs):
        super(MyWidget, self).__init__(**kwargs)
        self.drives_list.adapter.bind(on_selection_change=self.drive_selection_changed)

    def get_win_drives(self):
        if platform == 'win':
            import win32api

            drives = win32api.GetLogicalDriveStrings()
            drives = drives.split('\000')[:-1]

            return drives
        else:    
            return []

    def drive_selection_changed(self, *args):
        selected_item = args[0].selection[0].text
        self.file_chooser.path = selected_item

    def load(self,  path, filename, Fast):
        if Fast:
            afterloop(self.progress_bar)
        else:
            if filename:
                print 'user chose: %s' % os.path.join(path, filename[0])
                print path,
                print filename[0].split('\\')[-1]
                loop_fle(path, filename[0].split('\\')[-1], self.progress_bar)
            else:
                print 'user chose: %s' % path
                loop_dir(path, self.progress_bar)


class Loader(App):
    def build(self):
        return MyWidget()

if __name__ == '__main__':
    Loader().run()