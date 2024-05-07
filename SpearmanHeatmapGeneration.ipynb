import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from DataLoaderMain import melt_excel_data, load_variables
# Only import display if you are going to display individual DataFrames
# from IPython.display import display 

# Paths to the main AllVariablesAligned Excel file
AllVariablesAlignedExcel = r"xxx" # Specify the path to the 'AllVariablesAligned'file

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
ViscosityExcel = r"xxx" # Specify the path to the 'Viscosity'file
# Initialise the Viscosity table as Output_Feature_df
Output_Feature_df = pd.read_excel(ViscosityExcel)

# Create a dataframe for the combined table of Input_Features_df and Output_Feature_df as SupremeTable_df

SupremeTable_df = pd.merge(Input_Features_df, Output_Feature_df, how= 'outer' )

# Create a list to store the last rows Series for THE ENTIRE TIMESTEPS
last_rows_list = []

# Loop through each timestep from 1 to 7672
for timestep in range(1, 7673):
    # Filter the DataFrame for the current timestep
    timestep_df = SupremeTable_df[SupremeTable_df['Timestep'] == timestep]
    
    # Compute the correlation matrix for the current timestep
    correlation_matrix = timestep_df.drop(columns=['Timestep', 'Batch number']).corr(method='spearman')
    
    # Extract the last row of the correlation matrix
    last_row = correlation_matrix.iloc[:, -1]  # Assumes you want the last column
    
    # Give the Series a name corresponding to the timestep
    last_row.name = f'{timestep}'
    
    # Append the Series to the list
    last_rows_list.append(last_row)

# Concatenate all the Series in the list to form a DataFrame, with each Series becoming a column
last_rows_df = pd.concat(last_rows_list, axis=1)

plt.figure(figsize=(300, 150))
sns.heatmap(last_rows_df, annot=False, fmt='.2f', cmap='coolwarm', linewidths=.5, cbar_kws={"shrink": .5})
plt.tight_layout()
plt.show()
