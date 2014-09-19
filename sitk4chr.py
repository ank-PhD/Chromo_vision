__author__ = 'ank'

import SimpleITK as sitk
import matplotlib.pyplot as plt


def import_image():
    ImageRoot = "/home/ank/Documents/var"
    suffix = '/n4.jpeg'
    return sitk.ReadImage(ImageRoot+suffix)


def show_slice(img):
    slice = sitk.GetArrayFromImage(img)
    plt.imshow(slice, interpolation='nearest', cmap='gray')
    plt.colorbar()
    plt.show()


def erode_dilate(median_img):
    m = median_img > 20
    m = sitk.GrayscaleErode(m, 6)
    m = sitk.GrayscaleDilate(m, 2)
    return m

if __name__ == "__main__":
    img = sitk.VectorMagnitude(import_image())
    show_slice(img)
    median_img = sitk.Median(img)
    show_slice(median_img)
    mask = erode_dilate(median_img)
    show_slice(sitk.LabelOverlay(img, mask))