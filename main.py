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
import fitting



def main():
    fit = fitting.FIT()

    # Create file upload form for  measurement data and configuration
    st.sidebar.title("Impedance fitting")
    st.sidebar.header("Input Files")
    type = st.sidebar.radio(
        "File Types", ("IM3590", "FRA5095", "KFM2030"))
    measurement_files = st.sidebar.file_uploader("Measurement Files", accept_multiple_files=True)
    config_file = st.sidebar.file_uploader("Configure File")
    model = st.sidebar.radio("Loss Model", ("leastsq", "least_squares"))

    # Read data from measurement files
    freq_list = []
    z_measured_list = []
    if measurement_files is not None and config_file is not None:
        for measurement_file in measurement_files:
            if type == "IM3590":
                data = np.loadtxt(measurement_file, delimiter="\t")
                freq = data[:, 0]
                z_measured = data[:, 1] + 1j * data[:, 2]
            elif type == "FRA5095":
                data = np.loadtxt(measurement_file, delimiter=",", skiprows=1)
                freq = data[:, 1]
                z_measured = data[:, 4] + 1j * data[:, 5]
            elif type == "KFM2030":
                for i in range(100):
                    try:
                        data = np.loadtxt(measurement_file, skiprows=i, delimiter='\t')
                        break
                    except ValueError:
                        continue
                freq = data[:, 0]
                z_measured = data[:, 1] + 1j * data[:, 2]
            # 各データファイルからfreq, z_measuredをリストに格納
            freq_list.append(freq)
            z_measured_list.append(z_measured)
    
        # Read Configure File
        # Load initial parameters
        config = yaml.load(config_file.getvalue(), Loader=yaml.SafeLoader)

        # 
        st.sidebar.header("Parameters")



if __name__ == '__main__':
    main()


