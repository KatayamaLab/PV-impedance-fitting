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

        if 'initials' in st.session_state:
            initials = st.session_state['initials']
        else:
            initials = [param['initial'] for param in config['params']]

        for i, param in enumerate(config['params']):
            format = "%4.2e"

            st.sidebar.subheader(
                param['name'] + "(" + (param['unit'] if 'unit' in param else "-") + ")")

            # これスライダーから変更した方がよさそう
            # スライダーのバーは最小値と最大値に比例しないため、小さな数や大きな数を扱うのには不向きらしい
            initials[i] = st.sidebar.slider(
                label="Value",
                min_value=float(param['min']),
                max_value=float(param['max']),
                value=float(initials[i]),
                step=float(param['max']-param['min'])/100,
                key=param['name'],
                format=format
            )
            param_names.append(param['name'])
            param_units.append(param['unit'] if 'unit' in param else "-")
            param_lower_list.append(param['min'])
            param_upper_list.append(param['max'])

            param['max'] = st.sidebar.number_input(
                label="Upper", key=param['name'] + " max", value=float(param['max']), format=format)
            param['min'] = st.sidebar.number_input(
                label="Lower", key=param['name'] + " min", value=float(param['min']), format=format)



if __name__ == '__main__':
    main()


