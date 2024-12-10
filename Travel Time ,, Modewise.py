#!/usr/bin/env python
# coding: utf-8

# In[1]:


import openmatrix as om
import numpy as np
import os


# In[12]:


import os
import pandas as pd

def sum_csv_matrices(folder_path, output_file):
    """
    Sums all CSV files in a folder (OD matrix structure) and saves the result as a new CSV file.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        output_file (str): Path to save the resulting summed matrix.
    """
    # Get all CSV files in the folder
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    print(f"Found {len(csv_files)} CSV files. Summing them...")

    # Initialize a variable to store the summed matrix
    summed_matrix = None

    # Loop through each CSV file and add to the sum
    for file in csv_files:
        print(f"Processing file: {file}")
        # Read the CSV file, skipping the first row and first column
        df = pd.read_csv(file, index_col=0)

        # Convert to numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # Replace NaN values with 0
        df = df.fillna(0)

        # Initialize summed_matrix if it's None
        if summed_matrix is None:
            summed_matrix = df
        else:
            summed_matrix += df  # Add the current DataFrame to the summed matrix

    # Save the resulting summed matrix to a new CSV file
    summed_matrix.to_csv(output_file, index=True)
    print(f"Summed matrix saved to {output_file}")

# Example Usage
folder_path = "D:/Flood_Project/Model_running/Trial Run -4/Travel Time Anlysis/Matrisx_prep/Normal Period/PK-Total Matrix"  # Folder with CSV files
output_file = "D:/Flood_Project/Model_running/Trial Run -4/Travel Time Anlysis/Matrisx_prep/Normal Period/PK-Total Matrix/Summed_OP_Matrix.csv"  # Output file

sum_csv_matrices(folder_path, output_file)


# In[18]:


import os
import pandas as pd

def calculate_mode_specific_travel_time(total_time_file, total_trips_file, mode_trips_folder, output_folder):
    """
    Calculates mode-specific travel time matrices.

    Args:
        total_time_file (str): File path of the total travel time matrix (CSV/XLSX).
        total_trips_file (str): File path of the total trips matrix (CSV/XLSX).
        mode_trips_folder (str): Folder containing mode-specific trip number matrices (CSV/XLSX).
        output_folder (str): Folder to save the mode-specific travel time matrices.

    Returns:
        None
    """

    # Load total travel time matrix
    if total_time_file.endswith(".csv"):
        total_time = pd.read_csv(total_time_file, index_col=0)
    else:
        total_time = pd.read_excel(total_time_file, index_col=0)
    
    # Load total trips matrix
    if total_trips_file.endswith(".csv"):
        total_trips = pd.read_csv(total_trips_file, index_col=0)
    else:
        total_trips = pd.read_excel(total_trips_file, index_col=0)
    
    # Ensure indices match
    total_time.index = total_time.index.astype(str)
    total_time.columns = total_time.columns.astype(str)
    total_trips.index = total_trips.index.astype(str)
    total_trips.columns = total_trips.columns.astype(str)
    
    # Normalize matrices
    if not (total_time.shape == total_trips.shape):
        raise ValueError("Total travel time matrix and total trips matrix dimensions do not match!")

    # Process mode-specific trip matrices
    mode_files = [f for f in os.listdir(mode_trips_folder) if f.endswith(('.csv'))]

    for mode_file in mode_files:
        mode_file_path = os.path.join(mode_trips_folder, mode_file)
        
        # Load mode-specific trips matrix
        if mode_file.endswith(".csv"):
            mode_trips = pd.read_csv(mode_file_path, index_col=0)
        else:
            mode_trips = pd.read_excel(mode_file_path, index_col=0)
        
        # Ensure indices match
        mode_trips.index = mode_trips.index.astype(str)
        mode_trips.columns = mode_trips.columns.astype(str)

        if not (mode_trips.shape == total_time.shape):
            raise ValueError(f"Mode-specific trips matrix dimensions do not match for {mode_file}!")

        # Calculate mode-specific travel time
        mode_travel_time = total_time * (mode_trips / total_trips)
        
        # Save mode-specific travel time matrix
        output_file = os.path.join(output_folder, f"Mode_Travel_Time_{os.path.splitext(mode_file)[0]}.csv")
        mode_travel_time.to_csv(output_file)
        print(f"Saved mode-specific travel time for {mode_file} to {output_file}")

# Example Usage
if __name__ == "__main__":
    total_time_file = "D:/Flood_Project/Model_running/Trial Run -4/Travel Time Anlysis/Matrisx_prep/Normal Period/PK-Travel_time/PK_travel_time.csv"  # Replace with your file path
    total_trips_file = "D:/Flood_Project/Model_running/Trial Run -4/Travel Time Anlysis/Matrisx_prep/Normal Period/PK-Total Matrix/Summed_PK_Matrix.csv"      # Replace with your file path
    mode_trips_folder = "D:/Flood_Project/Model_running/Trial Run -4/Travel Time Anlysis/Matrisx_prep/Normal Period/PK-Mode_Matrix"  # Replace with folder path
    output_folder = "D:/Flood_Project/Model_running/Trial Run -4/Travel Time Anlysis/Matrisx_prep/Normal Period/PK-Mode_Matrix"  # Replace with folder path

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Calculate mode-specific travel times
    calculate_mode_specific_travel_time(total_time_file, total_trips_file, mode_trips_folder, output_folder)


# In[20]:


import pandas as pd

def filter_od_matrix(input_file, output_file, zones_to_keep):
    """
    Filters the OD matrix to keep only specific zones.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the filtered matrix.
        zones_to_keep (list): List of zone IDs to retain in the matrix.
    """
    # Read the matrix from the CSV file
    matrix = pd.read_csv(input_file, index_col=0)

    # Ensure the zone IDs in the matrix and zones_to_keep are of the same type
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    zones_to_keep = [str(zone) for zone in zones_to_keep]

    # Check for missing zones in the matrix
    missing_zones = [zone for zone in zones_to_keep if zone not in matrix.index or zone not in matrix.columns]
    if missing_zones:
        print(f"Warning: The following zones are not present in the matrix and will be ignored: {missing_zones}")

    # Filter rows and columns based on the provided zone list
    available_zones = [zone for zone in zones_to_keep if zone in matrix.index and zone in matrix.columns]
    filtered_matrix = matrix.loc[available_zones, available_zones]

    # Save the filtered matrix to a new CSV file
    filtered_matrix.to_csv(output_file)
    print(f"Filtered matrix saved to: {output_file}")


# Example Usage
# Example Usage
input_matrix_file = "D:/Flood_Project/Model_running/Trial Run -4/Travel Time Anlysis/Matrisx_prep/Normal Period/PK-Mode_Matrix/Mode_Travel_Time_NMT.csv"
output_filtered_matrix_file = "D:/Flood_Project/Model_running/Trial Run -4/Travel Time Anlysis/Matrisx_prep/Normal Period/PK-Mode_Matrix/Filterd_Mode_Travel_Time_NMT.csv"


# List of zones to keep
zones_to_keep = [
    52, 94, 95, 96, 97, 98, 100, 102, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 184, 186, 187, 189, 395, 396, 397, 398, 399, 400,
    401, 402, 403, 404, 405
]

filter_od_matrix(input_matrix_file, output_filtered_matrix_file, zones_to_keep)


# In[26]:


import pandas as pd
import numpy as np

def compute_accessibility_factor(travel_time_file, a_j_file, output_file, alpha=0.04):
    """
    Compute accessibility factors using the given travel time matrix and A_j table.

    Args:
        travel_time_file (str): Path to the CSV file containing the travel time matrix (T_ij).
        a_j_file (str): Path to the CSV file containing zone IDs and their corresponding A_j values.
        output_file (str): Path to save the resulting accessibility factors.
        alpha (float): The constant alpha to use in the calculation (default is 0.04).
    """
    # Load the travel time matrix
    travel_time_matrix = pd.read_csv(travel_time_file, index_col=0)
    travel_time_matrix = travel_time_matrix.apply(pd.to_numeric, errors='coerce')

    # Load the A_j table
    a_j_table = pd.read_csv(a_j_file)
    a_j_table = a_j_table.set_index('ZoneID')['Aj']  # Assuming columns: ZoneID, Aj

    # Initialize a dictionary to store accessibility factors
    accessibility_factors = {}

    # Compute exponential matrix: e^(-alpha * T_ij)
    exp_matrix = np.exp(-alpha * travel_time_matrix)

    # Multiply each column of the exp_matrix by the corresponding A_j value
    for zone_id in a_j_table.index:
        if zone_id in travel_time_matrix.columns:
            exp_matrix[zone_id] *= a_j_table[zone_id]

    # Compute the accessibility factor for each zone
    for zone_id in travel_time_matrix.index:
        if zone_id in exp_matrix.index:
            acc_fac = np.log(1 + exp_matrix.loc[zone_id].sum())
            accessibility_factors[zone_id] = acc_fac

    # Save the result to a CSV file
    result_df = pd.DataFrame.from_dict(accessibility_factors, orient='index', columns=['AccFac'])
    result_df.index.name = 'ZoneID'
    result_df.to_csv(output_file)

    print(f"Accessibility factors saved to: {output_file}")


# Example Usage
travel_time_csv = "D:/Flood_Project/Model_running/Trial Run -4/Travel Time Anlysis/Matrisx_prep/Flood/PK-Mode_Matrix/Filterd_Mode_Travel_Time_Auto.csv"  # Replace with your file path
a_j_csv = "D:/Flood_Project/Model_running/Trial Run -4/Acceibility Factor Transit/Flooding Scenario/PM/Attaction.csv"  # Replace with your file path
output_csv = "D:/Flood_Project/Model_running/Trial Run -4/Travel Time Anlysis/Matrisx_prep/Flood/Accesibility Analysis/Acc_Auto_PK_Matrix.csv"  # Replace with desired output file path

compute_accessibility_factor(travel_time_csv, a_j_csv, output_csv)

