import numpy as np
import pandas as pd
from datetime import datetime
import pytz

def nan_helper(y: np.ndarray) -> tuple[np.ndarray, callable]:
    """Helper to handle indices and logical indices of NaNs.

    Args:
        y (np.ndarray): 1d numpy array with possible NaNs

    Returns:
        tuple: Contains:
            - np.ndarray: logical indices of NaNs
            - callable: function with signature indices = index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices

    Examples:
        >>> # linear interpolation of NaNs
        >>> nans, x = nan_helper(y)
        >>> y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_nans(t: np.ndarray, y: np.ndarray, gap_dur: float = np.inf) -> np.ndarray:
    """Replace NaNs with interpolated values if the gap is shorter than gap_dur.

    Args:
        t (np.ndarray): Time array corresponding to y values
        y (np.ndarray): 1d numpy array with values to interpolate
        gap_dur (float, optional): Maximum duration of gap to interpolate in ms. Defaults to np.inf.

    Returns:
        np.ndarray: Array with interpolated values
    """
    # Find index for nans where gaps are longer than 'gap_dur' samples
    d = np.isnan(y)
    

    # If there are no nans, return
    if not np.any(d):
        return y

    # Find onsets and offsets of gaps
    d = np.diff(np.concatenate((np.array([0]), d*1, np.array([0]))))
    onsets = np.where(d==1)[0]
    offsets = np.where(d==-1)[0]
    # Decrease offsets come too late by -1
    if np.any(offsets >= len(y)):
        idx = np.where(offsets >= len(y))[0][0]
        offsets[idx] = offsets[idx] - 1

    dur = t[offsets] - t[onsets]


    # If the gaps are longer than 'gaps', replace temporarily with other values
    for i, on in enumerate(onsets):
        if dur[i] > gap_dur:
            y[onsets[i]:offsets[i]] = -1000

    nans, x = nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    # put nans back
    y[y == -1000] = np.nan

    return y

def nearest_odd_integer(x: float) -> int:
    """Return the nearest odd integer to the input value.

    Args:
        x (float): Input number to be converted

    Returns:
        int: Nearest odd integer

    Examples:
        >>> nearest_odd_integer(6.25)
        7
        >>> nearest_odd_integer(5.99)
        5
    """
    return int(2*np.floor(x/2)+1)



def compute_average_value(
    left: np.float64,
    right: np.float64
) -> np.float64:
    """Compute the average of two float values.

    Handles NaN values by using the valid value when one input is NaN.
    Returns NaN only when both inputs are NaN.

    Args:
        left (np.float64): First value to average
        right (np.float64): Second value to average

    Returns:
        np.float64: Average of the two values, or single valid value, or NaN
    """
    if not np.isnan(left) and not np.isnan(right):  # If both available, compute average
        return (left + right) / 2
    elif not np.isnan(left):  # If only left is available
        return left
    elif not np.isnan(right):  # If only right is available
        return right
    else:  # If neither is available, return NaN
        return np.nan


def compute_average_array(
    array1: np.ndarray,
    array2: np.ndarray
) -> np.ndarray:
    """Compute the element-wise average of two arrays.

    Args:
        array1 (np.ndarray): First array to average
        array2 (np.ndarray): Second array to average

    Returns:
        np.ndarray: Element-wise average of input arrays.
                   For each position, if one array contains NaN,
                   the valid value from the other array is kept.
                   If both arrays contain NaN at the same position,
                   the result remains NaN.
    """
    result = np.where(np.isnan(array1), array2,
                      np.where(np.isnan(array2), array1,
                      (array1 + array2) / 2))                                    
    return result

def extract_coordinates(
    column: str,
    df: pd.DataFrame
) -> pd.DataFrame:
    """Convert the list of 3D data of one eye into separate coordinate columns.

    Args:
        column (str): Name of the column containing coordinate string
        df (pd.DataFrame): DataFrame containing the coordinate data

    Returns:
        pd.DataFrame: DataFrame with three columns containing x, y, z coordinates as float values
    """
    return df[column].str.strip("()").str.split(",", expand=True).astype(float)

def preprocess_data(
    file_path: str
) -> pd.DataFrame:
    """Convert raw eye-tracking data into processed format for analysis.

    Processes the raw eye-tracking data by:
    - Converting timestamps to milliseconds
    - Extracting and averaging coordinates from both eyes
    - Computing gaze directions and positions
    - Preparing the dataset for blink and fixation analysis

    Args:
        file_path (str): Path to the raw eye-tracking data CSV file

    Returns:
        pd.DataFrame: Processed DataFrame containing:
            - Timestamps in milliseconds
            - Gaze coordinates and directions
            - Eye positions and movements
            - Validity indicators
            - Pupil diameter measurements
            - Eye openness values
    """
    # Read the raw data file and perform initial processing
    df = pd.read_csv(file_path)
    # Convert timestamps from microseconds to milliseconds and round
    df['recording_timestamp'] =  (df['system_time_stamp']/1000)
    df['recording_timestamp'] = df['recording_timestamp'] - df['recording_timestamp'][0]
    # Select subset of data rows and reset index
    spain_tz = pytz.timezone("Europe/Madrid")
    # datetime conversion code
    df['date_time'] = df['system_time_now'].apply(
        lambda x: datetime.fromtimestamp(x / 1000, spain_tz).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    )
    # Extract x,y coordinates for gaze points on display for both eyes
    df[["point_on_display_left_x", "point_on_display_left_y"]] = extract_coordinates("left_gaze_point_on_display_area",df)
    df[["point_on_display_right_x", "point_on_display_right_y"]] = extract_coordinates("right_gaze_point_on_display_area",df)

    # Extract x,y,z coordinates for gaze points and eye positions in user coordinate system
    df[["gaze_point_left_x", "gaze_point_left_y", "gaze_point_left_z"]] = extract_coordinates("left_gaze_point_in_user_coordinate_system",df)
    df[["eye_position_left_x", "eye_position_left_y", "eye_position_left_z"]] = extract_coordinates("left_gaze_origin_in_user_coordinate_system",df)

    df[["gaze_point_right_x", "gaze_point_right_y", "gaze_point_right_z"]] = extract_coordinates("right_gaze_point_in_user_coordinate_system",df)
    df[["eye_position_right_x", "eye_position_right_y", "eye_position_right_z"]] = extract_coordinates("right_gaze_origin_in_user_coordinate_system",df)

    # Get eye openness values
    df["eye_openness_left"] = df["left_eye_openness_value"]
    df["eye_openness_right"] = df["right_eye_openness_value"]

    # Classify eye movement type based on gaze origin validity
    df["eye_movement_type"] = df.apply(
        lambda row: "eyesnotfound" if (row["left_gaze_origin_validity"]==0) and (row["right_gaze_origin_validity"]==0) else "fixation",
        axis=1
    )

    # Define columns to keep in filtered dataset
    columns_to_keep = [
        "recording_timestamp",
        "date_time",
        "point_on_display_left_x",
        "point_on_display_left_y", 
        "point_on_display_right_x",
        "point_on_display_right_y",
        "gaze_point_left_x",
        "gaze_point_left_y",
        "gaze_point_left_z",
        "gaze_point_right_x",
        "gaze_point_right_y",
        "gaze_point_right_z",
        "eye_position_left_x",
        "eye_position_left_y",
        "eye_position_left_z",
        "eye_position_right_x",
        "eye_position_right_y",
        "eye_position_right_z",
        "left_gaze_origin_validity",
        "right_gaze_origin_validity",
        "left_pupil_diameter",
        "right_pupil_diameter",
        "eye_openness_left",
        "eye_openness_right",
        "eye_movement_type",
        "stimulus"
    ]
    df_filtered = df.copy()
    df_filtered = df_filtered[columns_to_keep]

    # Calculate gaze direction vectors for each eye by subtracting eye position from gaze point
    df_filtered['gaze_direction_left_x'] =df_filtered["gaze_point_left_x"]-df_filtered['eye_position_left_x']
    df_filtered['gaze_direction_left_y'] =df_filtered["gaze_point_left_y"]-df_filtered['eye_position_left_y']
    df_filtered['gaze_direction_left_z'] =df_filtered["gaze_point_left_z"]-df_filtered['eye_position_left_z']

    df_filtered['gaze_direction_right_x'] =df_filtered["gaze_point_right_x"]-df_filtered['eye_position_right_x']
    df_filtered['gaze_direction_right_y'] =df_filtered["gaze_point_right_y"]-df_filtered['eye_position_right_y']
    df_filtered['gaze_direction_right_z'] =df_filtered["gaze_point_right_z"]-df_filtered['eye_position_right_z']

    # Calculate average gaze direction across both eyes
    df_filtered["gaze_direction_x"] = df_filtered.apply(lambda row: compute_average_value(row["gaze_direction_left_x"], row["gaze_direction_right_x"]), axis=1)
    df_filtered["gaze_direction_y"] = df_filtered.apply(lambda row: compute_average_value(row["gaze_direction_left_y"], row["gaze_direction_right_y"]), axis=1)
    df_filtered["gaze_direction_z"] = df_filtered.apply(lambda row: compute_average_value(row["gaze_direction_left_z"], row["gaze_direction_right_z"]), axis=1)

    # Calculate average gaze point on display across both eyes
    df_filtered["point_on_display_x"] = df_filtered.apply(lambda row: compute_average_value(row["point_on_display_left_x"], row["point_on_display_right_x"]), axis=1)
    df_filtered["point_on_display_y"] = df_filtered.apply(lambda row: compute_average_value(row["point_on_display_left_y"], row["point_on_display_right_y"]), axis=1)

    # Calculate average eye position across both eyes
    df_filtered["eye_position_x"] = df_filtered.apply(lambda row: compute_average_value(row["eye_position_left_x"], row["eye_position_right_x"]), axis=1)
    df_filtered["eye_position_y"] = df_filtered.apply(lambda row: compute_average_value(row["eye_position_left_y"], row["eye_position_right_y"]), axis=1)
    df_filtered["eye_position_z"] = df_filtered.apply(lambda row: compute_average_value(row["eye_position_left_z"], row["eye_position_right_z"]), axis=1)
  
    # Calculate average gaze point in user coordinate system across both eyes
    df_filtered["gaze_point_x"] = df_filtered.apply(lambda row: compute_average_value(row["gaze_point_left_x"], row["gaze_point_right_x"]), axis=1)
    df_filtered["gaze_point_y"] = df_filtered.apply(lambda row: compute_average_value(row["gaze_point_left_y"], row["gaze_point_right_y"]), axis=1)
    df_filtered["gaze_point_z"] = df_filtered.apply(lambda row: compute_average_value(row["gaze_point_left_z"], row["gaze_point_right_z"]), axis=1)
    print(df_filtered.columns)
    return df_filtered
