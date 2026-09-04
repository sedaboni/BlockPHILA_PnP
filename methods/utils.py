import sys
from methods.iterators import *

def get_optimizer(optimizername):
    return getattr(sys.modules[__name__], optimizername)