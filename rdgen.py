__author__ = 'ank'

from random import shuffle
from matplotlib import pyplot as plt
from matplotlib import mlab as mlab
from scipy import sin, linspace


def generate_lab_list():
    Presenters = ['Guang Bo','Everett','Dominic','Andrei','Wahid', 'Parama', 'Tamara', 'Sree',
        'Pushpendra', 'Sam', 'Praveen','Hung-Ji', 'Pei-Shang', 'Yuping', 'Ben', 'Philip',
        'Kexi','Kai', 'Jin']
    shuffle(Presenters)
    print Presenters


def draw(stress_time, stress_level):
    plt.xkcd()
    plt.plot(5*mlab.normpdf(linspace(-10,100), 0, 3), color='blue')
    plt.plot(mlab.normpdf(linspace(-10,100), 20, 5), color='green')
    plt.plot(10*mlab.normpdf(linspace(-10,100), 30, 3), color='red')
    plt.show()


if __name__ == "__main__":
    draw(None, None)
    pass
