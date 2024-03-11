# -*- coding: utf-8 -*-

# Non-preinstalled packages
import streamlit as st
from scipy.optimize import leastsq, least_squares
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Default packages
from math import log10
import os
import datetime
import re
import builtins
import shutil


class Fitting:
    def __init__(self):
        self.errors = []
        self.error = False

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
        e = np.abs((z_ - z) / z_)
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
        local = {"func": None}

        # Define 'func' function as string
        exec_str = "def func(f, params):\n"
        for i, param in enumerate(params):
            exec_str += "  {} = params[{}]\n".format(param["name"], i)
        for d in defs:
            exec_str += "  " + d + "\n"
        exec_str += "  " + "return " + expr + "\n"

        # Define actual 'func' from string
        exec(exec_str, globals(), local)

        # Return defined 'func'
        return local["func"]

    def read_data(self, measurement_file, type="FRA5095"):
        voltages = []
        if type == "IM3590":
            data = np.loadtxt(measurement_file, delimiter="\t")
            freq = data[:, 0]
            z_measured = data[:, 1] + 1j * data[:, 2]

        elif type == "FRA5095":
            data = np.loadtxt(measurement_file, delimiter=",", skiprows=1)
            freq = data[:, 1]
            z_measured = data[:, 4] + 1j * data[:, 5]
            # set DC voltage
            voltages.append(data[1, 6])
            # DC voltage
            voltages.append(data[1, 7])
            # set AC voltage
            voltages.append(data[1, 8])
            # AC voltage
            voltages.append(data[1, 9])

        elif type == "KFM2030":
            for i in range(100):
                try:
                    data = np.loadtxt(measurement_file, skiprows=i, delimiter="\t")
                    break
                except ValueError:
                    continue
            freq = data[:, 0]
            z_measured = data[:, 1] + 1j * data[:, 2]

        elif type == "TG":
            data = np.loadtxt(measurement_file, delimiter=",")
            freq = data[:, 0]
            z_measured = data[:, 1] * np.cos(np.radians(data[:, 2])) + 1j * data[
                :, 1
            ] * np.sin(np.radians(data[:, 2]))

        return freq, z_measured, voltages

    @st.cache(hash_funcs={builtins.complex: lambda _: hash(abs(_))})
    def fit(
        self,
        initials,
        func,
        freq,
        z_measured,
        param_lowers,
        param_uppers,
        error_eval="absolute",
        model="leastsq",
    ):
        # Perform fitting and output results
        try:
            if error_eval == "absolute":
                if model == "leastsq":
                    param_result = leastsq(
                        self.loss_absolute, initials, args=(func, freq, z_measured)
                    )
                    loss = np.average(
                        self.loss_absolute(param_result[0], func, freq, z_measured)
                    )
                    param_list = param_result[0]
                elif model == "least_squares":
                    param_result = least_squares(
                        self.loss_absolute_least_squares,
                        initials,
                        args=(func, freq, z_measured),
                        bounds=(param_lowers, param_uppers),
                    )
                    loss = np.average(
                        self.loss_absolute(param_result.x, func, freq, z_measured)
                    )
                    param_list = param_result.x

            elif error_eval == "relative":
                if model == "leastsq":
                    param_result = leastsq(
                        self.loss_relative, initials, args=(func, freq, z_measured)
                    )
                    loss = np.average(
                        self.loss_absolute(param_result[0], func, freq, z_measured)
                    )
                    param_list = param_result[0]
                elif model == "least_squares":
                    param_result = least_squares(
                        self.loss_relative_least_squares,
                        initials,
                        args=(func, freq, z_measured),
                        bounds=(param_lowers, param_uppers),
                    )
                    loss = np.average(
                        self.loss_absolute(param_result.x, func, freq, z_measured)
                    )
                    param_list = param_result.x

            return param_list, loss

        except Exception as e:
            print(e)
            self.error = True
            self.errors.append("fit")
            return -1, -1

    def show_data(
        self,
        freq,
        z_measured=None,
        z_calc=None,
        param_names=None,
        param_values=None,
        param_units=None,
        loss=None,
    ):
        # Convert measured impedance data to pandas dataframe

        # Add calculated impedanca data to pandas dataframe
        if z_calc is None:
            df_graph = pd.DataFrame(
                {
                    "freq": freq,
                    "z_real": z_measured.real,
                    "z_imag": z_measured.imag,
                    "kind": "Measured",
                }
            )
            df_z = pd.DataFrame(
                {
                    "freq": freq,
                    "z_real_measured": z_measured.real,
                    "z_imag_measured": z_measured.imag,
                }
            )
        elif z_measured is None:
            df_graph = pd.DataFrame(
                {
                    "freq": freq,
                    "z_real": z_calc.real,
                    "z_imag": z_calc.imag,
                    "kind": "Fit",
                }
            )
            df_z = pd.DataFrame(
                {"freq": freq, "z_real_calc": z_calc.real, "z_imag_calc": z_calc.imag}
            )
        else:
            df_graph = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "freq": freq,
                            "z_real": z_measured.real,
                            "z_imag": z_measured.imag,
                            "kind": "Measured",
                        }
                    ),
                    pd.DataFrame(
                        {
                            "freq": freq,
                            "z_real": z_calc.real,
                            "z_imag": z_calc.imag,
                            "kind": "Fit",
                        }
                    ),
                ]
            )
            df_z = pd.DataFrame(
                {
                    "freq": freq,
                    "z_real_measured": z_measured.real,
                    "z_imag_measured": z_measured.imag,
                    "z_real_calc": z_calc.real,
                    "z_imag_calc": z_calc.imag,
                }
            )

        # Plot impedanca data
        fig = px.scatter(df_graph, x="z_real", y="z_imag", color="kind")
        fig["layout"]["yaxis"]["autorange"] = "reversed"
        fig["layout"]["yaxis"]["scaleanchor"] = "x"
        fig["layout"]["yaxis"]["title"] = "Z'' [Orms]"
        fig["layout"]["xaxis"]["title"] = "Z' [Orms]"
        st.plotly_chart(fig, use_container_width=True)

        # Display impedanca data
        st.dataframe(df_z)

        # Display parameters
        if param_names is not None or param_values is not None:
            data_text = "\n".join(
                [
                    "{}:\t{:.4g} {}".format(n, v, u)
                    for (n, v, u) in zip(param_names, param_values, param_units)
                ]
            )
            st.text(data_text)

        st.text("Mean square error: {}".format(loss))

    def save_temporary_data(
        self,
        dir_name,
        freq,
        z_measured,
        z_calc,
        param_names,
        param_values,
        param_units,
        config,
    ):
        # Remove prohibited characters from path name.
        dir_name = re.sub(r'[\\/:*?"<>|]+', " ", dir_name)

        # Make path to save temporary data
        result_dir = os.path.join("./temporary", dir_name)
        os.makedirs(result_dir, exist_ok=True)

        # Save data via pandas
        pd.DataFrame(
            {
                "freq": freq,
                "z_real_measured": z_measured.real,
                "z_imag_measured": z_measured.imag,
                "z_real_calc": z_calc.real,
                "z_imag_calc": z_calc.imag,
            }
        ).to_csv(os.path.join(result_dir, "impedance.csv"))

        pd.DataFrame(
            {"Name": param_names, "Value": param_values, "Unit": param_units}
        ).to_csv(os.path.join(result_dir, "parameters.csv"))

        # Save fitting condition as yaml
        with open(os.path.join(result_dir, "condition.yaml"), "wb") as f:
            f.write(yaml.dump(config, encoding="utf-8", allow_unicode=True))

        # Save graph by matplotlib
        z_measured_imag_fig = z_measured.imag * -1
        z_calc_imag_fig = z_calc.imag * -1
        fig = plt.figure(figsize=(16, 9))
        plt.scatter(
            z_measured.real,
            z_measured_imag_fig,
            s=50,
            marker="o",
            label="Measured",
            zorder=2,
        )
        plt.scatter(
            z_calc.real, z_calc_imag_fig, s=50, marker="x", label="Fit", zorder=2
        )
        plt.xlabel("$Z'$" + " [Ω]", fontsize=18)
        plt.ylabel("$-Z''$" + " [Ω]", fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left", frameon=False, fontsize=15)
        plt.grid(linestyle="--", zorder=1)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        fig.savefig(os.path.join(result_dir, "figure.png"))

    def save_all_parameters(self, all_parameter_values, param_names, type="FRA5095"):
        if type == "FRA5095":
            all_parameter_names = ["set_DC_volt", "DC_volt", "set_AC_volt", "AC_volt"]
            all_parameter_names.extend(list(param_names))

        else:
            all_parameter_names = list(param_names)

        all_parameter_names.append("loss")
        pd.DataFrame(
            data=np.array(all_parameter_values), columns=all_parameter_names
        ).to_csv(os.path.join("./temporary", "all_parameters.csv"))

    def save_data(self, comment, path_results):
        # Remove prohibited characters from path name.
        comment = re.sub(r'[\\/:*?"<>|]+', " ", comment)

        # Make path to save data
        result_dir = os.path.join(
            path_results,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + comment,
        )
        os.makedirs(result_dir, exist_ok=True)

        # Move result data from 'temporary' to 'results'
        for p in os.listdir("./temporary"):
            shutil.move(os.path.join("./temporary", p), result_dir)

    def get_freq(self, lower, upper, step=None):
        if lower <= 0:
            lower = 1e-8
        if upper < lower:
            upper = lower * 100

        if step is None:
            step = int(log10(upper) - log10(lower)) * 10 + 1
        freq = [
            10 ** (log10(lower) + (log10(upper) - log10(lower)) / (step - 1) * f)
            for f in range(step)
        ]
        return np.array(freq)
