__author__ = 'ank'

from chr_sep_human import human_loop as p_loop
from chr_sep_human import human_afterloop as p_afterloop
import os, errno
from pickle import load, dump
from kivy.clock import Clock
from mock import MagicMock, Mock


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def count_images(path):
    namelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)) and name.split('.')[-1] in ['jpeg', 'jpg', 'tif',' tiff']:
            namelist.append(name)
    return len(namelist)


def loop_dir(image_directory, widget):
    progress_bar, text_field = widget.progress_bar, widget.text_field
    progress_bar.value = 1
    cim =  count_images(image_directory)
    if not cim:
        t_to_add = 'Failed to find any .jpeg, .jpg, .tif or .tiff in the directory'
        text_field.text = text_field.text+t_to_add+'\n'
        progress_bar.value = 1000
        widget.append_to_consommables(t_to_add)
        return ''
    increment = 1000/cim
    afterloop_list = []
    buffer_directory = os.path.join(image_directory,'buffer')
    safe_mkdir(buffer_directory)
    for fle in os.listdir(image_directory):
        print fle
        prefix, suffix = ('_'.join(fle.split('.')[:-1]), fle.split('.')[-1])
        if suffix in ['jpeg', 'jpg', 'tif',' tiff']:
            buffer_path = os.path.join(buffer_directory, prefix)
            pre_time = p_loop(buffer_path, os.path.join(image_directory,fle), widget.stack_type)
            t_to_add = "file %s pre-processed in %s seconds" %(fle, "{0:.2f}".format(pre_time))
            afterloop_list.append((pre_time, prefix, buffer_path))
            progress_bar.value = progress_bar.value + increment
            widget.append_to_consommables(t_to_add)
    dump((image_directory, afterloop_list), open('DO_NOT_TOUCH.dmp','wb'))
    progress_bar.value = 1000
    return ''


def loop_fle(image_directory, file, widget):
    progress_bar, text_field = widget.progress_bar, widget.text_field
    progress_bar.value = 300
    widget.append_to_consommables('starting to process fle %s'%file)
    afterloop_list = []
    buffer_directory = os.path.join(image_directory,'buffer')
    safe_mkdir(buffer_directory)
    prefix, suffix = ('_'.join(file.split('.')[:-1]), file.split('.')[-1])
    if suffix in ['jpeg', 'jpg', 'tif',' tiff']:
        buffer_path = os.path.join(buffer_directory, prefix)
        print buffer_path
        safe_mkdir(buffer_path)
        pre_time = p_loop(buffer_path, os.path.join(image_directory, file), widget.stack_type)
        t_to_add = "file %s pre-processed in %s seconds" %(file, "{0:.2f}".format(pre_time))
        afterloop_list.append((pre_time, prefix, buffer_path))
    else:
        t_to_add = 'file %s has a wrong extension'%file
    widget.append_to_consommables(t_to_add)
    progress_bar.value = 1000
    dump((image_directory, afterloop_list), open('DO_NOT_TOUCH.dmp', 'wb'))
    return ''


def afterloop(widget):
    progress_bar, text_field = widget.progress_bar, widget.text_field
    imdir, afterloop_list = load(open('DO_NOT_TOUCH.dmp','rb'))
    output_directory = os.path.join(imdir, 'output')
    safe_mkdir(output_directory)
    for pre_time, fle_name, buffer_path in afterloop_list:
        t_to_add = p_afterloop(output_directory, pre_time, fle_name, buffer_path)
        text_field.text = text_field.text+t_to_add+'\n'


class progbar(object):
    def __init__(self):
        self.value = 0

class text_fields(object):
    def __init__(self):
        self.text = ''

class wdg(object):
    def __init__(self, st_tp):
        self.progress_bar = progbar()
        self.text_field = text_fields()
        self.stack_type = st_tp

if __name__ == "__main__":
    fname = 'img_000000002__000.tif'
    test_f = 'L:/Akn/mammalian/human chromosome spreads/10-7-14 rpe/rpe WT/rpe WT images 2'
    st_tp = 0
    loop_fle(test_f, fname, wdg(st_tp))
    afterloop(wdg(st_tp))
    pass
