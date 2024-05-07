# Import necessary packages and load SupremeTable.df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from functools import reduce
from DataLoaderMain import melt_excel_data, load_variables
# Only import display if you are going to display individual DataFrames
# from IPython.display import display 

# Paths to the main AllVariablesAligned Excel file
AllVariablesAlignedExcel = r"xxx" # Specify the path to the 'AllVariablesAligned' file

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
ViscosityExcel = r"xxx" # Specify the path to the 'Viscosity' file
# Initialise the Viscosity table as Output_Feature_df
Output_Feature_df = pd.read_excel(ViscosityExcel)

# Create a dataframe for the combined table of Input_Features_df and Output_Feature_df as SupremeTable_df

SupremeTable_df = pd.merge(Input_Features_df, Output_Feature_df, how= 'outer' )

# Specify the timestep to be analysed for the rest of the project, got from Spearman correlation analysis
MinTimeStep = 4200
MaxtimeStep = 5000

# Initialise the Shorten version of the SupremeTable, based on the specified time step above
Shorten_SupremeTable = SupremeTable_df.loc[(SupremeTable_df['Timestep'] >= MinTimeStep) & (SupremeTable_df['Timestep'] <= MaxtimeStep)]

# Remove Variable 11, 16, 22, based on the Spearman correlation analysis as well

interesting_columns = ['Batch number', 'Timestep', 'Variable1', 'Variable2', 'Variable3',
       'Variable4', 'Variable5', 'Variable6', 'Variable7', 'Variable8',
       'Variable9', 'Variable10', 'Variable12', 'Variable13',
       'Variable14', 'Variable15', 'Variable17', 'Variable18',
       'Variable19', 'Variable20', 'Variable21', 'Variable23',
       'Variable24', 'Variable25', 'Variable26', 'Variable27', 'Variable28', 'Viscosity (cP·s)']

final = Shorten_SupremeTable[interesting_columns]

# Initialise the Batches that will be strictly place in the training set from the Shorten_SupremeTable dataframe

val_set = [5,7,9,12,16,25,30]

training_df = final.loc[~final['Batch number'].isin(val_set)]
validation_df = final.loc[final['Batch number'].isin(val_set)]

''' THIS IS FOR THE TRAINING DATA ONLYYYY'''

# Number of features: assuming 'Batch number' and 'Timestep' are not part of the features
n_features = training_df.shape[1] - 3  # Subtract 'Batch number', 'Timestep', and the target column

# Find the maximum number of timesteps in any batch (if padding)
max_timesteps = training_df.groupby('Batch number').size().max()

# Initialize the 3D array for LSTM input
# For padding, use max_timesteps, otherwise use your fixed n_timesteps
n_batches = training_df['Batch number'].nunique()
input_data = np.zeros((n_batches, max_timesteps, n_features))

# Initialize an array for the target variable
target_data = np.zeros((n_batches,))

# Process each batch
for i, batch_number in enumerate(training_df['Batch number'].unique()):
    batch_data = training_df[training_df['Batch number'] == batch_number]
    
    # Assuming the 'Timestep' column is ordered
    # Extract features and target variable
    features = batch_data.drop(['Batch number', 'Timestep', 'Viscosity (cP·s)'], axis=1)
    target = batch_data['Viscosity (cP·s)'].iloc[-1]  # Assuming last viscosity value is the target
    
    # Pad or truncate
    padded_features = features.to_numpy()
    if len(features) > max_timesteps:
        padded_features = padded_features[:max_timesteps]  # Truncate
    else:
        # Pad with zeros if fewer timesteps than max_timesteps
        padded_features = np.pad(
            padded_features,
            ((0, max_timesteps - len(features)), (0, 0)),
            'constant',
            constant_values=0
        )
    
    # Assign to the appropriate slice of the input_data array
    input_data[i, :, :] = padded_features
    # Assign the target viscosity value
    target_data[i] = target

# Check the shape after padding or truncation
print(input_data.shape)
print(target_data.shape)

''' THIS IS FOR THE VALIDATION DATA ONLYYYY'''

# Number of features: assuming 'Batch number' and 'Timestep' are not part of the features
n_features_validation = validation_df.shape[1] - 3  # Subtract 'Batch number', 'Timestep', and the target column

# Find the maximum number of timesteps in any batch (if padding)
max_timesteps_validation = validation_df.groupby('Batch number').size().max()

# Initialize the 3D array for LSTM input
# For padding, use max_timesteps, otherwise use your fixed n_timesteps
n_batches_validation = validation_df['Batch number'].nunique()
input_data_validation = np.zeros((n_batches_validation, max_timesteps_validation, n_features_validation))

# Initialize an array for the target variable
target_data_validation = np.zeros((n_batches_validation,))

# Process each batch
for i, batch_number in enumerate(validation_df['Batch number'].unique()):
    batch_data_validation = validation_df[validation_df['Batch number'] == batch_number]
    
    # Assuming the 'Timestep' column is ordered
    # Extract features and target variable
    features_validation = batch_data_validation.drop(['Batch number', 'Timestep', 'Viscosity (cP·s)'], axis=1)
    target_validation = batch_data_validation['Viscosity (cP·s)'].iloc[-1]  # Assuming last viscosity value is the target
    
    # Pad or truncate
    padded_features_validation = features_validation.to_numpy()
    if len(features_validation) > max_timesteps_validation:
        padded_features_validation = padded_features_validation[:max_timesteps]  # Truncate
    else:
        # Pad with zeros if fewer timesteps than max_timesteps
        padded_features_validation = np.pad(
            padded_features_validation,
            ((0, max_timesteps_validation - len(features_validation)), (0, 0)),
            'constant',
            constant_values=0
        )
    
    # Assign to the appropriate slice of the input_data array
    input_data_validation[i, :, :] = padded_features_validation
    # Assign the target viscosity value
    target_data_validation[i] = target_validation

# Check the shape after padding or truncation
print(input_data_validation.shape)
print(target_data_validation.shape)

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adamax
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.layers import LSTM

# Reshape target_data and target_data_validation to have shape (n_samples, 1) if it's not already 2D
target_data = target_data.reshape(-1, 1)
target_data_validation = target_data_validation.reshape(-1, 1)

# Initialize the scaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# FOR THE TRAINING DATA ONLY!!!

# Fit the scaler on the input features and the target, and transform them
# We need to reshape the data to 2D to scale, then reshape back to 3D
n_samples, n_timesteps, n_features = input_data.shape
input_data_2d = input_data.reshape(-1, n_features)  # Reshape to 2D for scaling
scaled_input_data_2d = scaler_X.fit_transform(input_data_2d)
scaled_input_data = scaled_input_data_2d.reshape(n_samples, n_timesteps, n_features)  # Reshape back to 3D

scaled_target_data = scaler_y.fit_transform(target_data)

# The shape of your scaled input data
input_shape = scaled_input_data.shape[1:]  # (max_timesteps, n_features)

# FOR THE VALIDATION DATA ONLY!!!

# Fit the scaler on the input features and the target, and transform them
# We need to reshape the data to 2D to scale, then reshape back to 3D
n_samples_validation, n_timesteps_validation, n_features_validation = input_data_validation.shape
input_data_validation_2d = input_data_validation.reshape(-1, n_features_validation)  # Reshape to 2D for scaling
scaled_input_data_validation_2d = scaler_X.fit_transform(input_data_validation_2d)
scaled_input_data_validation = scaled_input_data_validation_2d.reshape(n_samples_validation, n_timesteps_validation, n_features_validation)  # Reshape back to 3D

scaled_target_data_validation = scaler_y.fit_transform(target_data_validation)

# The shape of your scaled input data
input_shape_validation = scaled_input_data_validation.shape[1:]  # (max_timesteps, n_features)

# Define the LSTM model
model = Sequential()

# Input layer
model.add(LSTM(units=27, input_shape=input_shape, return_sequences=False))

# Hidden layers
model.add(Dense(units=30, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(units=30, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(units=30, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(units=30, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(units=30, activation='relu', kernel_regularizer=l2(0.01)))


#kernel_regularizer=l2(0.01)

# Output layer
model.add(Dense(units=1))  # Single unit for a single output value per batch



# Compile the model
optimizer = SGD(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Fit the model
early_stopping_monitor = EarlyStopping(patience=20)
model_result = model.fit(
    scaled_input_data, scaled_target_data,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping_monitor]
)


# Make predictions using the unseen data

predictions = model.predict(scaled_input_data_validation)
predictions_real = scaler_y.inverse_transform(predictions)

# Inverse transform the scaled target data for validation


# Now let's evaluate the predictions of the above to the unseen data

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate MAE
mae = mean_absolute_error(scaled_target_data_validation, predictions)
print("Mean Absolute Error (MAE):", mae)

# Calculate MSE
mse = mean_squared_error(scaled_target_data_validation, predictions)
print("Mean Squared Error (MSE):", mse)

# Calculate RMSE
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate R-squared (R²)
r2 = r2_score(scaled_target_data_validation, predictions)
print("R-squared (R²):", r2)

#Model_Summary = pd.DataFrame(model_1_adam.summary())
Model_Loss_History = pd.DataFrame(model_result.history)
Simulation_Metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R2'],
    'Value': [mae, mse, rmse, r2]
})
# Create a DataFrame with actual and predicted values
Actual_v_Pred_df = pd.DataFrame({'Actual Values': scaled_target_data_validation.flatten(), 'Predicted Values': predictions.flatten()})
RealActual_v_RealPred_df = pd.DataFrame({'Actual Values': target_data_validation.flatten(), 'Predicted Values': predictions_real.flatten()})

