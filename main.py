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


def load_files():
    # Create file upload form for  measurement data and configuration
    st.sidebar.title("Impedance fitting")
    st.sidebar.header("Input Files")
    type = st.sidebar.radio(
        "File Types", ("IM3590", "FRA5095-v1", "FRA5095-v2", "KFM2030"))
    measurement_file = st.sidebar.file_uploader("Measurement File")
    config_file = st.sidebar.file_uploader("Configure File")
    fit_func = st.sidebar.radio(
        "Fitting Function", ("leastsq", "least_squares"))
    return measurement_file, type, config_file, fit_func


def read_data(measurement_file, type="FRA5095-v2"):
    if type == "IM3590":
        data = np.loadtxt(measurement_file, delimiter="\t")
        freq = data[:, 0]
        z_measured = data[:, 1] + 1j * data[:, 2]
    elif type == "FRA5095-v1":
        data = np.loadtxt(measurement_file, delimiter=",", skiprows=1)
        freq = data[:, 0]
        z_measured = data[:, 1]*(np.cos(data[:, 2] * np.pi / 180) +
                                 1j * np.sin(data[:, 2] * np.pi / 180))
    elif type == "FRA5095-v2":
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

    return freq, z_measured


def read_config(config_file):
    # Load initial parameters
    config = yaml.load(config_file.getvalue(), Loader=yaml.SafeLoader)
    return config


def draw_settings(config):
    # Set initial parameter and create slide bar
    st.sidebar.header("Parameters")

    if 'initials' in st.session_state:
        initials = st.session_state['initials']
    else:
        initials = [param['initial'] for param in config['params']]

    param_names = []
    param_units = []
    param_lower_list = []
    param_upper_list = []
    for i, param in enumerate(config['params']):
        format = "%4.2e"

        st.sidebar.subheader(
            param['name'] + "(" + (param['unit'] if 'unit' in param else "-") + ")")

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

    st.session_state['initials'] = initials

    st.sidebar.header("Other settings")
    config['fitting']['upper frequency'] = st.sidebar.number_input(
        "Upper frequency", value=float(config['fitting']['upper frequency']))
    config['fitting']['lower frequency'] = st.sidebar.number_input(
        "Lower frequency", value=float(config['fitting']['lower frequency']))

    # Error settings
    config['error evaluation'] = st.sidebar.selectbox(
        label="Error evaluation",
        options=["absolute", "relative"],
        index=["absolute", "relative"].index(config['error evaluation'])
    )
    return initials, param_names, param_units, param_lower_list, param_upper_list


@st.cache(hash_funcs={builtins.complex: lambda _: hash(abs(_))})
def fit(initials, func, freq, z_measured, param_lower_list, param_upper_list, error_eval="absolute", fit_func="leastsq"):
    # Perform fitting and output results
    if error_eval == "absolute":
        if fit_func == "leastsq":
            param_result = leastsq(loss_absolute, initials,
                                   args=(func, freq, z_measured))
            loss = np.average(loss_absolute(
                param_result[0], func, freq, z_measured))
            param_list = param_result[0]
        elif fit_func == "least_squares":
            param_result = least_squares(loss_absolute, initials,
                                         args=(func, freq, z_measured), bounds=(param_lower_list, param_upper_list))
            loss = np.average(loss_absolute(
                param_result.x, func, freq, z_measured))
            param_list = param_result.x
    elif error_eval == "relative":
        if fit_func == "leastsq":
            param_result = leastsq(loss_relative, initials,
                                   args=(func, freq, z_measured))
            loss = np.average(loss_absolute(
                param_result[0], func, freq, z_measured))
            param_list = param_result[0]
        elif fit_func == "least_squares":
            param_result = least_squares(loss_absolute, initials,
                                         args=(func, freq, z_measured), bounds=(param_lower_list, param_upper_list))
            loss = np.average(loss_absolute(
                param_result.x, func, freq, z_measured))
            param_list = param_result.x

    return param_list, loss


def show_data(freq, z_measured=None, z_calc=None, param_names=None, param_values=None, param_units=None, loss=None):
    # Convert measured impedance data to pandas dataframe

    # Add calculated impedanca data to pandas dataframe
    if z_calc is None:
        df_graph = pd.DataFrame(
            {"freq": freq, "z_real": z_measured.real, "z_imag": z_measured.imag, "kind": "Measured"})
        df_z = pd.DataFrame({
            "freq": freq,
            "z_real_measured": z_measured.real, "z_imag_measured": z_measured.imag,
        })
    elif z_measured is None:
        df_graph = pd.DataFrame(
            {"freq": freq, "z_real": z_calc.real, "z_imag": z_calc.imag, "kind": "Fit"})
        df_z = pd.DataFrame({
            "freq": freq,
            "z_real_calc": z_calc.real, "z_imag_calc": z_calc.imag
        })

    else:
        df_graph = pd.concat([
            pd.DataFrame(
                {"freq": freq, "z_real": z_measured.real, "z_imag": z_measured.imag, "kind": "Measured"}),
            pd.DataFrame(
                {"freq": freq, "z_real": z_calc.real, "z_imag": z_calc.imag, "kind": "Fit"})
        ])
        df_z = pd.DataFrame({
            "freq": freq,
            "z_real_measured": z_measured.real, "z_imag_measured": z_measured.imag,
            "z_real_calc": z_calc.real, "z_imag_calc": z_calc.imag
        })

    # Plot impedanca data
    fig = px.scatter(df_graph, x="z_real", y="z_imag", color="kind")
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig['layout']['yaxis']['scaleanchor'] = "x"
    fig['layout']['yaxis']['title'] = "Z'' [Orms]"
    fig['layout']['xaxis']['title'] = "Z' [Orms]"
    st.plotly_chart(fig, use_container_width=True)

    # Display impedanca data
    st.dataframe(df_z)

    # Display parameters
    if param_names is not None or param_values is not None:
        data_text = "\n".join(["{}:\t{:.4g} {}".format(n, v, u) for (
            n, v, u) in zip(param_names, param_values, param_units)])
        st.text(data_text)

    st.text("Mean square error: {}".format(loss))


def save_data(comment, freq, z_measured, z_calc,
              param_names, param_values, param_units, config):
    # Remove prohibited characters from path name.
    comment = re.sub(r'[\\/:*?"<>|]+', ' ', comment)

    # Make path to save data
    result_dir = os.path.join(
        "results",
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') +
        comment
    )
    os.makedirs(result_dir, exist_ok=True)

    # Save data via pandas
    pd.DataFrame(
        {"freq": freq,
         "z_real_measured": z_measured.real, "z_imag_measured": z_measured.imag,
         "z_real_calc": z_calc.real, "z_imag_calc": z_calc.imag}
    ).to_csv(os.path.join(result_dir, "impedance.csv"))

    pd.DataFrame({'Name': param_names, 'Value': param_values, 'Unit': param_units}
                 ).to_csv(os.path.join(result_dir, "parameters.csv"))

    # Save fitting condition as yaml
    with open(os.path.join(result_dir, "condition.yaml"), "wb") as f:
        f.write(
            yaml.dump(config, encoding="utf-8", allow_unicode=True))

    # Show finished
    st.write("Saved")

    zip_stream = io.BytesIO()

    with zipfile.ZipFile(zip_stream, 'w', compression=zipfile.ZIP_STORED) as new_zip:
        new_zip.write(os.path.join(result_dir, "impedance.csv"),
                      arcname='impedance.csv')
        new_zip.write(os.path.join(result_dir, "parameters.csv"),
                      arcname='parameters.csv')
        new_zip.write(os.path.join(result_dir, "condition.yaml"),
                      arcname='condition.yaml')

    return zip_stream.getvalue()


def get_download_link(zip_str):

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(zip_str).decode()
    return f'<a href="data:application/zip;base64,{b64}">Download zip file</a>'


def get_freq(lower, upper, step=None):
    if lower <= 0:
        lower = 1e-8
    if upper < lower:
        upper = lower*100

    if step is None:
        step = int(log10(upper) - log10(lower)) * 10 + 1
    freq = [10**(log10(lower)+(log10(upper) - log10(lower))/(step-1)*f)
            for f in range(step)]
    return np.array(freq)


def main():
    measurement_file, type, config_file, fit_func = load_files()

    if measurement_file is not None and config_file is not None:
        # if measurement and config file are both available, show measurement data and do fitting
        freq, z_measured = read_data(measurement_file, type)

        config = read_config(config_file)
        initials, param_names, param_units, param_lower_list, param_upper_list = draw_settings(
            config)

        func = define_func(config['params'], config['func']
                        ['defs'], config['func']['expr'])

        # Fitting!!!!!
        st.header("Results")
        if st.button("Fit!", help="Do Fitting!"):
            # Make limited measured data from upper/lower frequency
            freq_ = []
            z_measured_ = []
            for f, z in zip(freq, z_measured):
                if config['fitting']['lower frequency'] <= float(f) <= config['fitting']['upper frequency']:
                    freq_.append(f)
                    z_measured_.append(z)
            freq_ = np.array(freq_)
            z_measured_ = np.array(z_measured_)

            # Do fitting
            param_values, loss = fit(initials, func, freq_,
                                     z_measured_, param_lower_list, param_upper_list, config['error evaluation'], fit_func)

            # Update streamlit widget
            st.session_state.initials = param_values.tolist()
            # st.write(config, param_lower_list, param_upper_list)
        else:
            # Before fitting, use initial values as calc values
            param_values = initials
            loss = None

        # Calculate impedance using fit parameters
        z_calc = func(freq, param_values)

        show_data(freq, z_measured, z_calc, param_names,
                  param_values, param_units, loss)

        # Save button and saving data
        comment = st.text_input("Comment", value="")
        save = st.button("Save data (local version only)")
        download = st.button("Save and Download data")

        if save or download:
            if comment == "":
                st.write("Comment is required")
            else:
                config["comment"] = comment
                config["type"] = type
                zip_str = save_data(
                    comment, freq, z_measured, z_calc,
                    param_names, param_values, param_units, config)

                if download:
                    st.markdown(get_download_link(zip_str),
                                unsafe_allow_html=True)

    elif measurement_file is not None:
        # if only measurement file is available, show measurement data
        freq, z_measured = read_data(measurement_file, type)

        show_data(freq, z_measured)

    elif config_file is not None:
        # if only configuration file is available, show theoritical data
        config = read_config(config_file)
        param_values, param_names, param_units = draw_settings(config)

        func = define_func(config['params'], config['func']
                        ['defs'], config['func']['expr'])

        freq = get_freq(config['fitting']['lower frequency'],
                        config['fitting']['upper frequency'])

        # Calculate impedance using fit parameters
        z_calc = func(freq, param_values)

        show_data(freq=freq, z_calc=z_calc, param_names=param_names,
                  param_values=param_values, param_units=param_units)


if __name__ == '__main__':
    main()

# %%
