import numpy as np
import pandas as pd

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_nans(t, y, gap_dur=np.inf):
    ''' Replaces nans with interpolated values if the 'island' or nans/or gap
        is shorter than 'gap_dur'

    Args:
        y - 1d numpy array,
        gap_dur - duration of gap in ms

    Returns:
        y - interpolated array
    '''

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

def nearest_odd_integer(x):
    """Returns nearest odd integer of input value.

    Input:
        - x, float or integer
    Output:
        - integer value that is odd
    Example:
        >>> nearest_odd_integer(6.25)
            7
        >>> nearest_odd_integer(5.99)
            5
    """

    return int(2*np.floor(x/2)+1)



def compute_average_value(left, right):
    if not np.isnan(left) and not np.isnan(right):  # If both available, compute average
        return (left + right) / 2
    elif not np.isnan(left):  # If only left is available
        return left
    elif not np.isnan(right):  # If only right is available
        return right
    else:  # If neither is available, return NaN
        return np.nan


def compute_average_array(array1, array2):
    result = np.where(np.isnan(array1), array2,
                      np.where(np.isnan(array2), array1,
                      (array1 + array2) / 2))                                    
    return result
        
def filter_data(file_path):
    df = pd.read_csv(file_path)
    selected_columns = ["Recording timestamp",
                        "Eye position left X", "Eye position right X",
                        "Eye position left Y", "Eye position right Y",
                        "Eye position left Z", "Eye position right Z",
                        "Gaze point left X", "Gaze point right X",
                        "Gaze point left Y", "Gaze point right Y",
                        "Gaze point left Z", "Gaze point right Z",
                        "Eye openness left", "Eye openness right",
                        "Eye movement type"]
    df_filtered = df[selected_columns].copy()
    '''
    selected_columns = ["Recording timestamp",
                        "Gaze direction left X", "Gaze direction left Y", "Gaze direction left Z",
                        "Gaze direction right X", "Gaze direction right Y", "Gaze direction right Z",
                        "Eye position left X (DACSmm)", "Eye position right X (DACSmm)",
                        "Eye position left Y (DACSmm)", "Eye position right Y (DACSmm)",
                        "Eye position left Z (DACSmm)", "Eye position right Z (DACSmm)",
                        "Gaze point left X (DACSmm)", "Gaze point right X (DACSmm)",
                        "Gaze point left Y (DACSmm)", "Gaze point right Y (DACSmm)",
                        "Gaze point left X", "Gaze point left Y",
                        "Gaze point right X", "Gaze point right Y",
                        "Eye openness left","Eye openness right","Eye movement type"]
    df_filtered = df[selected_columns].copy()
    '''
    df_filtered["Recording timestamp"] = df_filtered["Recording timestamp"]/1000
    df_filtered["Recording timestamp"] = df_filtered["Recording timestamp"].round(0)

    '''
    df_filtered["Recording timestamp"] = np.where(
        df_filtered["Recording timestamp"] % 10 == 9,  # Condition: Last 2 digits == 99
        df_filtered["Recording timestamp"] + 1,  # If true, add 1 to round up
        df_filtered["Recording timestamp"] ) # If false, keep as is

    df_filtered["Recording timestamp"] = np.where(
        df_filtered["Recording timestamp"] % 10 == 1,
        df_filtered["Recording timestamp"]  -1,
        df_filtered["Recording timestamp"] )
    '''
    # Compute average gaze direction, eye position and gaze point
    df_filtered['Gaze direction left X'] =df_filtered["Gaze point left X"]-df_filtered['Eye position left X']
    df_filtered['Gaze direction left Y'] =df_filtered["Gaze point left Y"]-df_filtered['Eye position left Y']
    df_filtered['Gaze direction left Z'] =df_filtered["Gaze point left Z"]-df_filtered['Eye position left Z']

    df_filtered['Gaze direction right X'] =df_filtered["Gaze point right X"]-df_filtered['Eye position right X']
    df_filtered['Gaze direction right Y'] =df_filtered["Gaze point right Y"]-df_filtered['Eye position right Y']
    df_filtered['Gaze direction right Z'] =df_filtered["Gaze point right Z"]-df_filtered['Eye position right Z']
    

    df_filtered["Gaze direction X"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze direction left X"], row["Gaze direction right X"]), axis=1)
    df_filtered["Gaze direction Y"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze direction left Y"], row["Gaze direction right Y"]), axis=1)
    df_filtered["Gaze direction Z"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze direction left Z"], row["Gaze direction right Z"]), axis=1)



    df_filtered["Eye position X"] = df_filtered.apply(lambda row: compute_average_value(row["Eye position left X"], row["Eye position right X"]), axis=1)
    df_filtered["Eye position Y"] = df_filtered.apply(lambda row: compute_average_value(row["Eye position left Y"], row["Eye position right Y"]), axis=1)
    df_filtered["Eye position Z"] = df_filtered.apply(lambda row: compute_average_value(row["Eye position left Z"], row["Eye position right Z"]), axis=1)
    df_filtered["Gaze point X"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze point left X"], row["Gaze point right X"]), axis=1)
    df_filtered["Gaze point Y"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze point left Y"], row["Gaze point right Y"]), axis=1)
    df_filtered["Gaze point Z"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze point left Z"], row["Gaze point right Z"]), axis=1)
    

    '''
    df_filtered["Gaze direction X"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze direction left X"], row["Gaze direction right X"]), axis=1)
    df_filtered["Gaze direction Y"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze direction left Y"], row["Gaze direction right Y"]), axis=1)
    df_filtered["Gaze direction Z"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze direction left Z"], row["Gaze direction right Z"]), axis=1)
    df_filtered["Eye position X"] = df_filtered.apply(lambda row: compute_average_value(row["Eye position left X (DACSmm)"], row["Eye position right X (DACSmm)"]), axis=1)
    df_filtered["Eye position Y"] = df_filtered.apply(lambda row: compute_average_value(row["Eye position left Y (DACSmm)"], row["Eye position right Y (DACSmm)"]), axis=1)
    df_filtered["Eye position Z"] = df_filtered.apply(lambda row: compute_average_value(row["Eye position left Z (DACSmm)"], row["Eye position right Z (DACSmm)"]), axis=1)
    
    df_filtered["Gaze point X (DACSmm)"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze point left X (DACSmm)"], row["Gaze point right X (DACSmm)"]), axis=1)
    df_filtered["Gaze point Y (DACSmm)"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze point left Y (DACSmm)"], row["Gaze point right Y (DACSmm)"]), axis=1)
    df_filtered["Gaze point X"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze point left X"], row["Gaze point right X"]), axis=1)
    df_filtered["Gaze point Y"] = df_filtered.apply(lambda row: compute_average_value(row["Gaze point left Y"], row["Gaze point right Y"]), axis=1)
    '''
    
    return df_filtered

   
   
'''
file_path = "csv results/custom_raw.xlsx" 
df = pd.read_excel(file_path)
selected_columns = ["Recording timestamp","Eye openness left","Eye openness right","Eye movement type"]
df= df[~df['Event'].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
f = df[selected_columns].copy()
# compute average direction and openness of both eyes
df["Eye openness"] = df.apply(lambda row: compute_gaze_direction(row["Eye openness left"], row["Eye openness right"]), axis=1)
df.loc[df["Eye movement type"] == "EyesNotFound", "Eye openness"] = np.nan
df = df.iloc[5:].reset_index(drop=True)
eo_signal = df["Eye openness"].values
time = df["Recording timestamp"].values
print(eo_signal)
sol = interpolate_nans(time,eo_signal)
'''