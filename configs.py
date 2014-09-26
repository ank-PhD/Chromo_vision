__author__ = 'ank'

from chr_sep_human import human_loop as p_loop
from chr_sep_human import human_afterloop as p_afterloop
import os, errno
from pickle import load, dump


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def count_images(path):
    return len([name for name in os.listdir('.')
                if os.path.isfile(name)
                    and name.split('.')[-1] in ['jpeg', 'jpg', 'tif',' tiff']])


def loop_dir(image_directory, progress_bar, text_field):
    progress_bar.value = 1
    cim =  count_images(image_directory)
    if not cim:
        t_to_add = 'Failed to find any .jpeg, .jpg, .tif or .tiff in the directory'
        text_field.text = text_field.text+t_to_add+'\n'
        progress_bar.value = 1000
        return ''
    increment = 1000/cim
    afterloop_list = []
    buffer_directory = os.path.join(image_directory,'buffer')
    safe_mkdir(buffer_directory)
    for fle in os.listdir(image_directory):
        prefix, suffix = ('_'.join(fle.split('.')[:-1]), fle.split('.')[-1])
        if suffix in ['jpeg', 'jpg', 'tif',' tiff']:
            buffer_path = os.path.join(buffer_directory, prefix)
            print buffer_path
            safe_mkdir(buffer_path)
            pre_time = p_loop(buffer_path, image_directory+fle)
            t_to_add = "file %s pre-processed in %s seconds" %(file, "{0:.2f}".format(pre_time))
            afterloop_list.append((pre_time, prefix, buffer_path))
            progress_bar.value = progress_bar.value + increment
            text_field.text = text_field.text+t_to_add+'\n'
    dump((image_directory, afterloop_list), open('DO_NOT_TOUCH.dmp','wb'))
    progress_bar.value = 1000
    return ''

def loop_fle(image_directory, file, progress_bar, text_field):
    progress_bar.value = 500
    afterloop_list = []
    buffer_directory = os.path.join(image_directory,'buffer')
    safe_mkdir(buffer_directory)
    prefix, suffix = ('_'.join(file.split('.')[:-1]), file.split('.')[-1])
    if suffix in ['jpeg', 'jpg', 'tif',' tiff']:
        buffer_path = os.path.join(buffer_directory, prefix)
        print buffer_path
        safe_mkdir(buffer_path)
        pre_time = p_loop(buffer_path, os.path.join(image_directory, file))
        t_to_add = "file %s pre-processed in %s seconds" %(file, "{0:.2f}".format(pre_time))
        afterloop_list.append((pre_time, prefix, buffer_path))
    else:
        t_to_add = 'file %s has a wrong extension'%file
    text_field.text = text_field.text+t_to_add+'\n'
    dump((image_directory, afterloop_list), open('DO_NOT_TOUCH.dmp', 'wb'))
    progress_bar.value = 1000
    return ''

def afterloop(progress_bar, text_field):
    imdir, afterloop_list = load(open('DO_NOT_TOUCH.dmp','rb'))
    output_directory = os.path.join(imdir, 'output')
    safe_mkdir(output_directory)
    for pre_time, fle_name, buffer_path in afterloop_list:
        t_to_add = p_afterloop(output_directory, pre_time, fle_name, buffer_path)
        text_field.text = text_field.text+t_to_add+'\n'

if __name__ == "__main__":
    print 0
    pass
