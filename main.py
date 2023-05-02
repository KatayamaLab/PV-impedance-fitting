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
        st_ = st if section_key == "general" else st.sidebar
        if section_key != "params":
            st_.header(section['title'])
            for item in section['items']:
                format = "%4.2e"
                if item['type'] == "number":
                    config[section_key][item['name']] = st_.number_input(
                        label=item['label'],
                        min_value=float(item['min']),
                        max_value=float(item['max']),
                        value=float(config[section_key][item['name']]),
                        format=format,
                        help=item['help']
                    )

                elif item['type'] == "string":
                    config[section_key][item['name']] = st_.text_input(
                        label=item['label'],
                        value=config[section_key][item['name']],
                        help=item['help']
                    )

                elif item['type'] == "selection":
                    config[section_key][item['name']] = st_.selectbox(
                        label=item['label'],
                        options=item['options'],
                        index=item['options']
                        .index(config[section_key][item['name']]),
                        help=item['help']
                    )
                    opt = item['options'].index(config[section_key][item['name']])

        elif section_key == "params":
            # Set initial parameter and create slide bar
            st_.header(section['title'])

            param_names = []
            initials = []
            param_lowers = []
            param_uppers = []
            param_units = []
            for param in config['params']:
                format = "%4.2e"
                st_.subheader(
                    param['name'] + "(" + (param['unit'] if 'unit' in param else "-") + ")")
                param['initial'] = st_.number_input(
                    label="Value",
                    min_value=float(param['min']),
                    max_value=float(param['max']),
                    value=float(param['initial']),
                    step=float(param['max']-param['min'])/100,
                    key=param['name'],
                    format=format,
                    help=item['help']
                )
                param['max'] = st_.number_input(
                    label="Upper", key=param['name'] + " max", value=float(param['max']), format=format)
                param['min'] = st_.number_input(
                    label="Lower", key=param['name'] + " min", value=float(param['min']), format=format)
                param_names.append(param['name'])
                initials.append(param['initial'])
                param_lowers.append(param['min'])
                param_uppers.append(param['max'])
                param_units.append(param['unit'] if 'unit' in param else "-")
    
    # Set 'general' setting
    type = config['general']['file types']
    model = config['fitting']['loss model']
    error_eval = config['fitting']['error evaluation']

    # Set equivalent circuit function
    func = fit.define_func(config['params'],
                        config['func']['defs'],
                        config['func']['expr'])


    if measurement_files:

        # フィッティング結果ごとのディレクトリ名を入力
        dir_names = []
        for i in range(len(measurement_files)):
            dir_name = st.text_input(
                label=str(i+1) + "つ目のリザルトディレクトリ名",
                max_chars=50,
                value="result" + str(i+1),
                help="リザルトディレクトリ名"
            )
            dir_names.append(dir_name)

        st.header("Results")

        # Fitting
        if st.button("Fit!", help="Do Fitting!"):

            initials_temp = np.empty(0) # 空のndarray配列を生成
            all_parameters = [] # 空のlistを生成
            for (measurement_file, dir_name) in zip(measurement_files, dir_names):
                # Read data from measurement files
                freq, z_measured, voltages = fit.read_data(measurement_file, type)

                freq_ = []
                z_measured_ = []
                for f, z in zip(freq, z_measured):
                    if config['fitting']['lower frequency'] <= float(f) <= config['fitting']['upper frequency']:
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

                # Calculate impedance using fitted parameters
                z_calc = func(freq, param_values)

                if dir_name == "":
                    st.write("Directory name is required")
                else:
                    config['general']['directory name'] = dir_name
                    fit.save_temporary_data(dir_name, freq, z_measured, z_calc,
                                            param_names, param_values, param_units, config)
                    
                    # フィッティングパラメータの統合ファイルを生成
                    voltages.extend(list(param_values))
                    all_parameters.append(voltages)
                    fit.save_all_parameters(all_parameters, param_names)
            
            # Show last data
            fit.show_data(freq, z_measured, z_calc, param_names,
                        param_values, param_units, loss)
        
        else:
            param_values = initials
            loss = None
        
            # Show first data
            freq, z_measured, voltages = fit.read_data(measurement_files[0], type)
            z_calc = func(freq, param_values)
            fit.show_data(freq, z_measured, z_calc, param_names,
                        param_values, param_units, loss)

        # Save button and saving data
        comment = st.text_input("Comment", value="")
        save = st.button("Save data")
        if save:
            if comment == "":
                st.write("Comment is required")
            else:
                path_results = config['general']['result path']
                fit.move_data(comment, path_results)

            # test
            # print(config)
            # print(freq_list)
            # print(z_measured_list)
            # print(freq_)
            # print(z_measured_)
            # print(initials)

    elif config_file is not None:
        # if only configuration file is available, show theoritical data
        freq = fit.get_freq(config['fitting']['lower frequency'],
                            config['fitting']['upper frequency'])

        # Calculate impedance using fit parameters
        z_calc = func(freq, initials)

        # Show data
        fit.show_data(freq=freq, z_calc=z_calc, param_names=param_names,
                  param_values=initials, param_units=param_units)


if __name__ == '__main__':
    main()


