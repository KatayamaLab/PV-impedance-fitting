# -*- coding: utf-8 -*-

# Non-preinstalled package
from numpy.testing._private.utils import measure
import streamlit as st
from scipy.optimize import leastsq, least_squares, curve_fit
import plotly.express as px
import numpy as np
import pandas as pd
import yaml
from math import log10


# Default packages
import os
import datetime
import re
import builtins
import base64
import zipfile
import io

class FIT():
    def __init__(self):
        self.errors = []
        self.error = False
    
    def __del__(self):
        pass

    def loss_absolute(param, func, f, z):
        z_ = func(f, param)
        e = np.abs(z_ - z)
        return e

    def loss_relative(param, func, f, z):
        z_ = func(f, param)
        e = np.abs((z_ - z)/z_)
        return e
    
    def define_func(params, defs, expr):
        # Use 'func' as local namespace
        local = {'func': None}

        # Define 'func' function as string
        exec_str = "def func(f, params):\n"
        for i, param in enumerate(params):
            exec_str += "  {} = params[{}]\n".format(param['name'], i)
        for d in defs:
            exec_str += "  " + d + "\n"
        exec_str += "  " + "return " + expr + "\n"

        # Define actual 'func' from string
        exec(exec_str, globals(), local)

        # Return defined 'func'
        return local['func']