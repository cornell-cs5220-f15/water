import sys
from math import *

def round(x):
    y = int(x)
    return (y/20)*20


if __name__=='__main__':
    p = int(sys.argv[1])
    print round(400*p**(1/3.0))

