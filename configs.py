__author__ = 'ank'

from chr_sep_human import human_loop as p_loop
from chr_sep_human import human_afterloop as p_afterloop
import os, errno
from pickle import load, dump

# image_directory = 'U:/ank/2014/Image_recognition/'
# buffer_directory = 'L:/ank/buffer/'
# image_to_load = 'img_19jpg.jpeg'

# image_directory = '/home/ank/projects_files/2014/Image_recognition/sources/'
# buffer_directory = '/home/ank/projects_files/2014/Image_recognition/buffer/'
# output_directory = '/home/ank/projects_files/2014/Image_recognition/outs/'
# image_to_load =  'img_12jpg.jpeg'

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


def loop_dir(image_directory, progress_bar):
    progress_bar.value = 1
    increment = 1000/count_images(image_directory)
    afterloop_list = []
    buffer_directory = os.path.join(image_directory,'buffer')
    safe_mkdir(buffer_directory)
    for fle in os.listdir(image_directory):
        prefix, suffix = ('_'.join(fle.split('.')[:-1]), fle.split('.')[-1])
        if suffix in ['jpeg', 'jpg', 'tif',' tiff']:
            buffer_path = os.path.join(buffer_directory, prefix)+'/'
            safe_mkdir(buffer_path)
            pre_time = p_loop(buffer_path, image_directory+fle)
            afterloop_list.append((pre_time, prefix, buffer_path))
            progress_bar.value = progress_bar.value + increment

    dump((image_directory, afterloop_list), open('DO_NOT_TOUCH.dmp','wb'))
    progress_bar.value = 1000


def loop_fle(image_directory, file, progress_bar):
    progress_bar.value = 500
    afterloop_list = []
    buffer_directory = os.path.join(image_directory,'buffer')
    safe_mkdir(buffer_directory)
    prefix, suffix = ('_'.join(file.split('.')[:-1]), file.split('.')[-1])
    if suffix in ['jpeg', 'jpg', 'tif',' tiff']:
        buffer_path = buffer_directory+prefix+'/'
        safe_mkdir(buffer_path)
        pre_time = p_loop(buffer_path, os.path.join(image_directory,file))
        afterloop_list.append((pre_time, prefix, buffer_path))
    dump((image_directory, afterloop_list), open('DO_NOT_TOUCH.dmp','wb'))
    progress_bar.value = 1000


def afterloop(progress_bar):
    imdir, afterloop_list = load(open('DO_NOT_TOUCH.dmp','rb'))
    output_directory = os.path.join(imdir, 'output')
    safe_mkdir(output_directory)
    for pre_time, fle_name, buffer_path in afterloop_list:
        p_afterloop(output_directory, pre_time, fle_name, buffer_path)


if __name__ == "__main__":
    print 0
    pass
