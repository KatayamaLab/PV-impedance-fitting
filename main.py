# -*- coding: utf-8 -*-

# Non-preinstalled packages
import streamlit as st
from scipy.optimize import leastsq, least_squares
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import fitting

# Default packages
from math import log10
import os
import datetime
import re
import builtins
import base64
import zipfile
import io


def main():
    fit = fitting.FIT()

    # Create file upload form for  measurement data and configuration
    st.sidebar.title("Impedance fitting")
    st.sidebar.header("Input Files")
    measurement_files = st.sidebar.file_uploader("Measurement Files", accept_multiple_files=True)
    config_file = st.sidebar.file_uploader("Configure File")

    # Load configulation file
    with open("config_template.yaml", "r", encoding="utf-8") as f:
        config_template = yaml.load(f, Loader=yaml.SafeLoader)

    with open("config_default.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if config_file is not None:
        config = yaml.load(config_file.getvalue(), Loader=yaml.SafeLoader)

    for (section_key, section) in zip(config_template.keys(), config_template.values()):
        if section_key == "general":
            st.sidebar.header(section['title'])
            for item in section['items']:
                format = "%4.2e"
                if item['type'] == "number":
                    config[section_key][item['name']] = st.sidebar.number_input(
                        label=item['label'],
                        min_value=float(item['min']),
                        max_value=float(item['max']),
                        value=float(config[section_key][item['name']]),
                        format=format,
                        help=item['help']
                    )

                elif item['type'] == "string":
                    config[section_key][item['name']] = st.sidebar.text_input(
                        label=item['label'],
                        value=config[section_key][item['name']],
                        help=item['help']
                    )

                elif item['type'] == "selection":
                    config[section_key][item['name']] = st.sidebar.selectbox(
                        label=item['label'],
                        options=item['options'],
                        index=item['options']
                        .index(config[section_key][item['name']]),
                        help=item['help']
                    )
                    opt = item['options'].index(config[section_key][item['name']])

        elif section_key == "params":
            # Set initial parameter and create slide bar
            st.sidebar.header(section['title'])

            param_names = []
            initials = []
            param_lowers = []
            param_uppers = []
            param_units = []
            for param in config['params']:
                format = "%4.2e"
                st.sidebar.subheader(
                    param['name'] + "(" + (param['unit'] if 'unit' in param else "-") + ")")
                param['initial'] = st.sidebar.number_input(
                    label="Value",
                    min_value=float(param['min']),
                    max_value=float(param['max']),
                    value=float(param['initial']),
                    step=float(param['max']-param['min'])/100,
                    key=param['name'],
                    format=format,
                    help=item['help']
                )
                param['max'] = st.sidebar.number_input(
                    label="Upper", key=param['name'] + " max", value=float(param['max']), format=format)
                param['min'] = st.sidebar.number_input(
                    label="Lower", key=param['name'] + " min", value=float(param['min']), format=format)
                param_names.append(param['name'])
                initials.append(param['initial'])
                param_lowers.append(param['min'])
                param_uppers.append(param['max'])
                param_units.append(param['unit'] if 'unit' in param else "-")
    
    # Set 'general' setting
    type = config['general']['file types']
    model = config['general']['loss model']
    error_eval = config['general']['error evaluation']

    # Set equivalent circuit function
    func = fit.define_func(config['params'],
                        config['func']['defs'],
                        config['func']['expr'])

    st.header("Results")

    if measurement_files:
        # Fitting
        if st.button("Fit!", help="Do Fitting!"):
            
            initials_temp = np.empty(0) # 空のndarray配列を生成
            for measurement_file in measurement_files:
                # Read data from measurement files
                freq, z_measured = fit.read_data(measurement_file, type)

                freq_ = []
                z_measured_ = []
                for f, z in zip(freq, z_measured):
                    if config['general']['lower frequency'] <= float(f) <= config['general']['upper frequency']:
                        freq_.append(f)
                        z_measured_.append(z)
                    else:
                        pass    # 指定範囲外のfとzは除外
                freq_ = np.array(freq_)
                z_measured_ = np.array(z_measured_)

                # 一つ前のフィッティング結果を再利用する
                if initials_temp.size != 0:
                    initials = initials_temp
                else:
                    pass

                # Do fitting
                param_values, loss = fit.fit(initials, func, freq_, z_measured_,
                                                param_lowers, param_uppers, error_eval, model)

                # フィッティング結果を再利用するためにinitials_tempに一旦格納
                initials_temp = param_values
                loss_temp = loss    # lossが消えるのを防止
            
            # Show last data
            z_calc = func(freq, param_values)
            fit.show_data(freq, z_measured, z_calc, param_names,
                        param_values, param_units, loss)
        
        else:
            param_values = initials
            loss = None
        
            # Show first data
            freq, z_measured = fit.read_data(measurement_files[0], type)
            z_calc = func(freq, param_values)
            fit.show_data(freq, z_measured, z_calc, param_names,
                        param_values, param_units, loss)
            

            # test
            # print(config)
            # print(freq_list)
            # print(z_measured_list)
            # print(freq_)
            # print(z_measured_)
            # print(initials)

    elif config_file is not None:
        # if only configuration file is available, show theoritical data
        freq = fit.get_freq(config['general']['lower frequency'],
                            config['general']['upper frequency'])

        # Calculate impedance using fit parameters
        z_calc = func(freq, initials)

        # Show data
        fit.show_data(freq=freq, z_calc=z_calc, param_names=param_names,
                  param_values=initials, param_units=param_units)


if __name__ == '__main__':
    main()


