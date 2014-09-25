__author__ = 'ank'

from chr_sep_human import human_loop, human_afterloop
import os, errno


image_directory = 'U:/ank/2014/Image_recognition/'
buffer_directory = 'L:/ank/buffer/'
image_to_load = 'img_19jpg.jpeg'


image_directory = '/home/ank/projects_files/2014/Image_recognition/sources/'
buffer_directory = '/home/ank/projects_files/2014/Image_recognition/buffer/'
output_directory = '/home/ank/projects_files/2014/Image_recognition/outs/'
image_to_load =  'img_12jpg.jpeg'



if __name__ == "__main__":
    afterloop_list = []
    for fle in os.listdir(image_directory):
        prefix, suffix = ('_'.join(fle.split('.')[:-1]), fle.split('.')[-1])
        if suffix in ['jpeg', 'jpg', 'tif',' tiff']:
            buffer_path = buffer_directory+prefix+'/'
            try:
                os.mkdir(buffer_path)
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(buffer_path):
                    pass
                else: raise
            # slow loading
            pre_time = human_loop(buffer_path, image_directory+fle)
            afterloop_list.append((pre_time, prefix, buffer_path))

    # wait for input
    raw_input("Please manually edit the mask image, save it. Once you are done, press enter to continue ")

    # fast lane
    for pre_time, fle_name, buffer_path in afterloop_list:
        human_afterloop(output_directory, pre_time, fle_name, buffer_path)

    print 'Done!'
