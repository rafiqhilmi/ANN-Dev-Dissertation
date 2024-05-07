import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from DataLoaderMain import melt_excel_data, load_variables
# Only import display if you are going to display individual DataFrames
# from IPython.display import display 

# Paths to the main AllVariablesAligned Excel file
AllVariablesAlignedExcel = r"xxx" # Specify the path to the raw data file 'AllVariablesAlligned' here

Big_Data_Melted = melt_excel_data(AllVariablesAlignedExcel)
Individual_Variables = load_variables(Big_Data_Melted)

# Initialize merged DataFrame with an empty DataFrame
Input_Features_df = pd.DataFrame()

# Loop through, set index and merge in one go
for sensor in range(1, 29):
    variable_key = f'Variable{sensor}'
    df = Individual_Variables.get(variable_key)
    if df is not None:
        df = df.set_index(['Batch number', 'Timestep'])
        Input_Features_df = pd.merge(Input_Features_df, df, left_index=True, right_index=True, how='outer', suffixes=('', f'_Sensor_{sensor}')) if not Input_Features_df.empty else df

# Reset index if you want 'Batch number' and 'Timestep' back as columns
Input_Features_df.reset_index(inplace=True)

# merged_df now contains all sensors' data merged on 'Batch number' and 'Timestep'

# Path to the output feature Excel file (Viscosity)
ViscosityExcel = r"xxx" # Specify the path to the raw data file 'Viscosity' here

# Initialise the Viscosity table as Output_Feature_df
Output_Feature_df = pd.read_excel(ViscosityExcel)

# Create a dataframe for the combined table of Input_Features_df and Output_Feature_df as SupremeTable_df

SupremeTable_df = pd.merge(Input_Features_df, Output_Feature_df, how= 'outer' )

x_min, x_max = SupremeTable_df['Timestep'].min(), SupremeTable_df['Timestep'].max()

# Assuming you're plotting for a specific 'variable_key' value
# Loop through each 'variable_key' if needed
for variable_key in range(1, 29):  # Adjust this range as necessary
    variable_name = f'Variable{variable_key}'  # Construct the column name dynamically

    plt.figure(figsize=(20, 8))

    for batch in range(1, 31):  # Loop through each batch
        process_batch = SupremeTable_df[SupremeTable_df['Batch number'] == batch]
        x = process_batch['Timestep']
        y = process_batch[variable_name]  # Use the dynamically constructed column name

        plt.plot(x, y, linestyle='-', label=f'{batch}')

    # Move plotting configurations outside the batch loop if you want one plot per 'variable_key'
    plt.title(f'Sensor {variable_key} vs. Timestep')
    plt.xlabel('Timestep')
    plt.xticks(np.arange(start=x_min, stop=x_max + 1, step=500))
    plt.ylabel(f'Sensor {variable_key}')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1), borderaxespad=0., title= 'Batch')
    plt.grid(True)
    plt.show()  # Consider moving this outside the loops if you want a single plot
