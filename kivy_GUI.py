from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
from kivy.utils import platform
from configs import loop_dir, loop_fle, afterloop
from kivy.properties import ObjectProperty

import os
from time import sleep

class MyWidget(BoxLayout):
    progress_bar = ProgressBar(max=1000)
    text_field = TextInput()

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
            afterloop(self.progress_bar, self.text_field)
            t_to_add = 'will try to post-process files pre-processed since the previous >>>'
            self.text_field.text = self.text_field.text+t_to_add+'\n'
            self.text_field._update_graphics()
            sleep(0.5)
        else:
            if filename:
                t_to_add = '>>> will try to pre-process file %s at %s'%(filename, path)
                self.text_field.text = self.text_field.text+t_to_add+'\n'
                self.text_field._update_graphics()
                sleep(0.5)
                loop_fle(path, filename[0].split('\\')[-1], self.progress_bar, self.text_field)
            else:
                t_to_add = '>>> will try to pre-process all images at %s'%path
                self.text_field.text = self.text_field.text+t_to_add+'\n'
                self.text_field._update_graphics()
                sleep(0.5)
                loop_dir(path, self.progress_bar, self.text_field)


class Loader(App):
    def build(self):
        return MyWidget()

if __name__ == '__main__':
    Loader().run()