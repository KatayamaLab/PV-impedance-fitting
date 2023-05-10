# -*- coding: utf-8 -*-

# Non-preinstalled packages
import streamlit as st
import yaml
import numpy as np
import fitting


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

    # Draw configulation
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

        elif section_key == "params":
            # Set initial parameter and create side bar input forms
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
        # Depending on the number of files uploaded, generate an input form 
        # for the directory name to store the individual fitting results.
        dir_names = []
        for i in range(len(measurement_files)):
            dir_name = st.text_input(
                label= "Result directory " + str(i+1),
                max_chars=50,
                value="result" + str(i+1),
                help="フィッティング結果ごとのディレクトリ名に当たります。"
            )
            dir_names.append(dir_name)

        st.header("Results")

        # Fitting
        if st.button("Fit!", help="Do Fitting!"):

            initials_temp = np.empty(0) # Create a empty ndarray
            all_parameter_values = [] # Create a empty list

            for (measurement_file, dir_name) in zip(measurement_files, dir_names):

                # Read data from measurement files
                freq, z_measured, voltages = fit.read_data(measurement_file, type)

                # Set frequency range and z_measured range
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

                # Use last fitting results, if any.
                if initials_temp.size != 0:
                    initials = initials_temp
                else:
                    pass

                # Do fitting
                param_values, loss = fit.fit(initials, func, freq_, z_measured_,
                                                param_lowers, param_uppers, error_eval, model)

                # Keep fitting results temporarily.
                initials_temp = param_values
                st.session_state['params'] = param_values
                st.session_state['loss'] = loss

                # Calculate impedance using fitted parameters
                z_calc = func(freq, param_values)

                # Store data in a 'temporary' directory
                if dir_name == "":
                    st.write("Directory name is required")
                else:
                    config['general']['directory name'] = dir_name
                    fit.save_temporary_data(dir_name, freq, z_measured, z_calc,
                                            param_names, param_values, param_units, config)
                    
                    # Generate a csv file with all parameters organized
                    voltages.extend(list(param_values))
                    all_parameter_values_ = voltages
                    all_parameter_values.append(all_parameter_values_)
                    fit.save_all_parameters(all_parameter_values, param_names)
            
            # Show last data
            fit.show_data(freq, z_measured, z_calc, param_names,
                        param_values, param_units, loss)
        
        else:
            if 'params' in st.session_state:
                param_values = st.session_state['params']
                show_file = measurement_files[-1]
            else:
                param_values = initials
                show_file = measurement_files[0]

            if 'loss' in st.session_state:
                loss = st.session_state['loss']
            else:
                loss = None
                
            # Show first data
            freq, z_measured, voltages = fit.read_data(show_file, type)
            z_calc = func(freq, param_values)
            fit.show_data(freq, z_measured, z_calc, param_names,
                        param_values, param_units, loss)

        # "Save data" button and saving data
        comment = st.text_input("Comment", value="")
        save = st.button("Save data")
        if save:
            if comment == "":
                st.write("Comment is required")
            else:
                path_results = config['general']['result path']
                fit.save_data(comment, path_results)
                st.write("Saving is success!")

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


