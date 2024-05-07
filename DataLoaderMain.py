import pandas as pd

def melt_excel_data(features_path):
    """
    Reads an allVariablesAligned Excel file from the specified path, melts the data for each sheet that matches
    the pattern 'VariableX', and returns a dictionary of melted DataFrames.

    Parameters:
    features_path (str): Path to the Excel file containing the data.

    Returns:
    dict: A dictionary where keys are variable names and values are the corresponding melted DataFrames.
    """
    # Read all sheets (tabs) in the Excel file
    all_variables_aligned = pd.read_excel(features_path, sheet_name=None, header=None)

    # Dictionary to store the melted DataFrames
    big_data_melted = {}

    for tab_name, var_n_tab in all_variables_aligned.items():
        # Check if tab_name matches the required variables
        if tab_name.startswith("Variable") and int(tab_name.replace("Variable", "")) in range(1, 29):
            # Directly insert 'Batch number' as the first column with appropriate values
            var_n_tab.insert(0, 'Batch number', range(1, len(var_n_tab) + 1))

            # Unpivot the dataframe
            var_n_melted = pd.melt(var_n_tab, id_vars=['Batch number'], var_name='Timestep', value_name=tab_name)

            # Store the melted DataFrame directly in the dictionary
            big_data_melted[tab_name] = var_n_melted
    
    return big_data_melted

def load_variables(big_data_melted):

    '''
    Used to load the individual sensor data (VariableX, where X is 1-28) stored in the big_data_melted dictionary.

    '''
    variable_dfs = {}
    for i in range(1, 29):
        variable_name = f'Variable{i}'
        if variable_name in big_data_melted:
            variable_dfs[variable_name] = big_data_melted[variable_name]
    return variable_dfs
