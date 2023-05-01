# -*- coding: utf-8 -*-

# Non-preinstalled packages
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
        self.data = []
    
    def __del__(self):
        pass
    
    # Define loss function(absolute)
    def loss_absolute(self, param, func, f, z):
        z_ = func(f, param)
        e = np.abs(z_ - z)
        # print(e)
        return e

    # Define loss function(relative)
    def loss_relative(self, param, func, f, z):
        z_ = func(f, param)
        e = np.abs((z_ - z)/z_)
        return e
    
    # Define loss function(absolute)
    def loss_absolute_least_squares(self, param, func, f, z):
        z_ = func(f, param)
        e = np.abs(z_ - z) * np.sqrt(2)
        return e

    # Define loss function(relative)
    def loss_relative_least_squares(self, param, func, f, z):
        z_ = func(f, param)
        e = np.abs(z_ - z) * np.sqrt(2)
        return e
    
    # Define fitting model
    def define_func(self, params, defs, expr):
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
    
    def read_data(self, measurement_file, type="FRA5095"):
        data = self.data
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

        return freq, z_measured

    @st.cache(hash_funcs={builtins.complex: lambda _: hash(abs(_))})
    def fit(self, initials, func, freq, z_measured, param_lowers, param_uppers, error_eval="absolute", model="leastsq"):
        # Perform fitting and output results
        try:
            if error_eval == "absolute":
                if model == "leastsq":
                    param_result = leastsq(self.loss_absolute, initials,
                                        args=(func, freq, z_measured))
                    loss = np.average(self.loss_absolute(param_result[0],
                                                    func, freq, z_measured))
                    param_list = param_result[0]
                elif model == "least_squares":
                    param_result = least_squares(self.loss_absolute_least_squares, initials,
                                                args=(func, freq, z_measured), bounds=(param_lowers, param_uppers))
                    loss = np.average(self.loss_absolute_least_squares(param_result.x,
                                                                func, freq, z_measured))
                    param_list = param_result.x

            elif error_eval == "relative":
                if model == "leastsq":
                    param_result = leastsq(self.loss_relative, initials,
                                        args=(func, freq, z_measured))
                    loss = np.average(self.loss_absolute(param_result[0],
                                                    func, freq, z_measured))
                    param_list = param_result[0]
                elif model == "least_squares":
                    param_result = least_squares(self.loss_relative_least_squares, initials,
                                                args=(func, freq, z_measured), bounds=(param_lowers, param_uppers))
                    loss = np.average(self.loss_absolute_least_squares(param_result.x,
                                                                func, freq, z_measured))
                    param_list = param_result.x
        
            return param_list, loss
        
        except Exception as e:
            print(e)
            self.error = True
            self.errors.append("fit")
            return -1, -1


    def show_data(self, freq, z_measured=None, z_calc=None, param_names=None, param_values=None, param_units=None, loss=None):
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

