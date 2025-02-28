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



def compute_gaze_direction(left, right):
    if not np.isnan(left) and not np.isnan(right):  # If both available, compute average
        return (left + right) / 2
    elif not np.isnan(left):  # If only left is available
        return left
    elif not np.isnan(right):  # If only right is available
        return right
    else:  # If neither is available, return NaN
        return np.nan


def combina_array_nan(array1, array2):
    result = np.where(np.isnan(array1), array2,
                      np.where(np.isnan(array2), array1,
                      (array1 + array2) / 2))                                    
    return result
        
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