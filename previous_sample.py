import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
import numpy as np
import pandas as pd
import scipy.signal as signal
import numpy as np
from scipy.signal import savgol_filter, find_peaks

#################################################
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
#%%
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


#%%
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
#########################################


def interpolate_data_loss(eo_signal, time):
    """
    Interpolate segments of data loss in the EO signal if the segment is shorter than max_gap_duration (in ms).
    """
    # Convert max_gap_duration from ms to samples
    # Find where the signal is NaN (data loss)
    nan_indices = np.where(np.isnan(eo_signal))[0]
    # Group consecutive NaN indices
    gaps = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0] + 1)
    # Interpolate gaps shorter than max_gap_samples
    for gap in gaps:
        start = max(0, gap[0] - 1)
        end = min(len(eo_signal) - 1, gap[-1] + 1)
        eo_signal = eo_signal.copy()
        for id in gap:
            print(id)
            t1 = time.iloc[id]
            t2 = time.iloc[start]
            t3 = time.iloc[end]
            scalin_fac = (t2-t3)/(t1-t3)
            new = scalin_fac * eo_signal.iloc[end]
            new += eo_signal.iloc[start]
            eo_signal.iloc[id] = new
    return eo_signal
def interpolate_missing_data(EO_signal, time):
    """Interpolate missing EO signal values for gaps smaller than max_gap ms."""
    valid = ~np.isnan(EO_signal)
    EO_signal_interp = np.interp(time, time[valid], EO_signal[valid])
    return EO_signal_interp

def smooth_signal(EO_signal, window_length=25, polyorder=2):
    """Apply Savitzky–Golay filter to smooth the EO signal."""
    return signal.savgol_filter(EO_signal, window_length, polyorder)


def detect_blinks(eo_signal, fs=60, savgol_window=25, vel_threshold_factor=3, min_amplitude=50, min_duration=50):
    """
    Detect blinks in the EO signal using the described algorithm.
    """
    # Step 1: Interpolate data loss
    eo_signal = interpolate_data_loss(eo_signal, fs=fs)
    
    # Step 2: Low-pass filter the EO signal with a Savitzky-Golay filter
    savgol_window_samples = int(savgol_window * fs / 1000)
    if savgol_window_samples >= len(eo_signal):
        raise ValueError("Savitzky-Golay window size is larger than the signal length.")
    eo_filtered = savgol_filter(eo_signal, savgol_window_samples, 2)
    
    # Step 3: Find peaks in the EO signal
    peaks, _ = find_peaks(eo_filtered, height=min_amplitude)
    
    # If no peaks are found, return an empty list
    if len(peaks) == 0:
        return []
    
    # Step 4: Compute velocity of the unfiltered EO signal
    velocity = savgol_filter(eo_signal, savgol_window_samples, 2, deriv=1)
    
    # Step 5: Define blink candidates using velocity threshold
    mad_velocity = np.median(np.abs(velocity - np.median(velocity)))
    vel_threshold = vel_threshold_factor * mad_velocity
    
    blink_candidates = []
    for peak in peaks:
        # Go backward in time to find onset
        onset = peak
        while onset > 0 and velocity[onset] > vel_threshold:
            onset -= 1
        
        # Go forward in time to find offset
        offset = peak
        while offset < len(velocity) - 1 and velocity[offset] > vel_threshold:
            offset += 1
        
        blink_candidates.append((onset, peak, offset))
    
    # Step 6: Compute peak opening and closing velocities
    blink_stats = []
    for onset, peak, offset in blink_candidates:
        peak_opening_velocity = np.max(velocity[onset:peak])
        peak_closing_velocity = np.max(velocity[peak:offset])
        amplitude = eo_filtered[peak] - np.min(eo_filtered[onset:offset])
        duration = (offset - onset) * 1000 / fs  # Convert to ms
        
        blink_stats.append({
            'onset': onset,
            'peak': peak,
            'offset': offset,
            'opening_velocity': peak_opening_velocity,
            'closing_velocity': peak_closing_velocity,
            'amplitude': amplitude,
            'duration': duration
        })
    
    # Step 7: Reject blink candidates based on thresholds
    valid_blinks = []
    for stat in blink_stats:
        if (stat['opening_velocity'] > vel_threshold and
            stat['closing_velocity'] > vel_threshold and
            stat['amplitude'] >= min_amplitude and
            stat['duration'] >= min_duration):
            valid_blinks.append(stat)
    
    return valid_blinks
    


def moving_median(df):
    global fix
    df_filtered = df.copy()
    for i in range(len(fix)):
        if i == 0 or i == len(fix)-1:
            continue
        id = fix[i]
        prev = id-1
        post = id+1
        window = df.loc[prev:post]
        if not window.isna().values.any():
            for col in df.columns:
                if (col != "Eye movement type") and (col != "Recording timestamp") and (col != "Eye openness"):
                    median = np.median([df.iloc[prev][col],df.iloc[id][col],df.iloc[post][col]])
                    df_filtered.at[id,col] = median
    return df_filtered


def compute_gaze_direction(left, right):
    if not np.isnan(left) and not np.isnan(right):  # If both available, compute average
        return (left + right) / 2
    elif not np.isnan(left):  # If only left is available
        return left
    elif not np.isnan(right):  # If only right is available
        return right
    else:  # If neither is available, return NaN
        return np.nan
    
def compute_average_interval(samples, num_samples=100):
    """
    Calcola l'intervallo medio (in secondi) tra i campioni 
    usando i primi num_samples campioni (o meno se non disponibili).
    """
    n = num_samples
    intervals = [samples.loc[i+1,"Recording timestamp"] - samples.loc[i,"Recording timestamp"] for i in range(n - 1)]
    return np.mean(intervals)

def compute_velocity(df):
    global fix, WINDOW_LENGTH
    df['Velocity'] = np.nan  # Initialize velocity column
    # Iterate through fixation indexes
    for i in range(len(fix)):
        if i == 0 :
            continue  # Skip first and last fixations
        else:
            index_start = fix[i-1]
            index_end = fix[i]
            sample_start = df.iloc[index_start]
            sample_end = df.iloc[index_end]
            dt = (sample_end['Recording timestamp'] - sample_start['Recording timestamp']) / 1000  # Convert ms to sec
            direction_start = np.array(sample_start[['Gaze direction X', 'Gaze direction Y', 'Gaze direction Z']])
            direction_end = np.array(sample_end[['Gaze direction X', 'Gaze direction Y', 'Gaze direction Z']])
            dot = np.dot(direction_start, direction_end)
            norm_start = np.linalg.norm(direction_start)
            norm_end   = np.linalg.norm(direction_end)
            if norm_end == 0 or norm_start == 0:
                continue
            theta = np.arccos(np.clip(dot / (norm_start * norm_end),-1,1))
            theta_deg = np.degrees(theta)
            ang_vel = theta_deg / dt
            df.at[index_end, 'Velocity'] = ang_vel
    return df

# Function to classify eye movements
def classify_movements(df):
    df['Classified Movement'] = df['Eye movement type']  # Preserve existing labels

    for i in range(len(fix)):
        idx = fix[i]

        if i == 0:
            prev_idx = None
            prev_move = "None"
        else:
            prev_idx = fix[i-1]
            prev_move = df.iloc[idx-1]['Classified Movement']

        if pd.isna(df.iloc[idx]['Velocity']):  # If velocity couldn't be computed, keep existing label
            continue
        
        if df.iloc[idx]['Velocity'] < VELOCITY_THRESHOLD:
            curr_move = "Fixation"
        else:
            curr_move = "Saccade"

        if prev_move in ["EyesNotFound", "Blink"]:
            curr_move = df.iloc[prev_idx]['Classified Movement']
        
        df.at[idx, 'Classified Movement'] = curr_move

    return df


def merge_adjacent_fixations(df):
    fixation_groups = []
    current_fixation = []

    for i in range(len(df)):
        if df.iloc[i]['Classified Movement'] == 'Fixation':
            if not current_fixation:
                current_fixation = [i]  # Start a new fixation group
            else:
                prev_idx = current_fixation[-1]
                time_gap = df.iloc[i]['Recording timestamp'] - df.iloc[prev_idx]['Recording timestamp']

                # Extract gaze direction vectors (normalized unit vectors)
                G1 = np.array(df.iloc[prev_idx][['Gaze direction X', 'Gaze direction Y', 'Gaze direction Z']])
                G2 = np.array(df.iloc[i][['Gaze direction X', 'Gaze direction Y', 'Gaze direction Z']])

                # Compute the dot product
                dot_product = np.dot(G1, G2)
                G1_norm = np.linalg.norm(G1)
                G2_norm = np.linalg.norm(G2)
                # Clamp dot product to avoid numerical errors (ensuring it's between -1 and 1)

                # Compute visual angle in degrees using arccos
                visual_angle = np.degrees(np.arccos(dot_product/(G1_norm * G2_norm)))

                # Merge fixations if time gap and visual angle conditions are met
                if time_gap <= MAX_TIME_BETWEEN_FIXATIONS and visual_angle <= MAX_ANGLE_BETWEEN_FIXATIONS:
                    current_fixation.append(i)
                else:
                    fixation_groups.append(current_fixation)
                    current_fixation = [i]  # Start new fixation group
        else:
            if current_fixation:
                fixation_groups.append(current_fixation)
                current_fixation = []

    if current_fixation:
        fixation_groups.append(current_fixation)

    # Assign merged fixation labels
    for group in fixation_groups:
        for idx in group:
            df.at[idx, 'Classified Movement'] = 'Fixation'
    return df



# Function to discard short fixations
def discard_short_fixations(df):
    start_idx = None
    for i in range(len(df)):
        if df.iloc[i]['Classified Movement'] == 'Fixation':
            if start_idx is None:
                start_idx = i  # Start of a fixation
        else:
            if start_idx is not None:
                duration = df.iloc[i - 1]['Recording timestamp'] - df.iloc[start_idx]['Recording timestamp']
                if duration < MIN_FIXATION_DURATION:
                    for j in range(start_idx, i):
                        df.at[j, 'Classified Movement'] = 'Unclassified'
                start_idx = None

    return df


# Main function to process eye-tracking data
if __name__ == "__main__":
    file_path = "csv results/custom_raw.xlsx" 
    df = pd.read_excel(file_path)
    IVT_df = pd.read_excel("csv results/noise_ivt.xlsx")
    selected_columns = ["Recording timestamp",
                        "Gaze direction left X", "Gaze direction left Y", "Gaze direction left Z",
                        "Gaze direction right X", "Gaze direction right Y", "Gaze direction right Z",
                        "Eye openness left","Eye openness right","Eye movement type"]
    df= df[~df['Event'].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
    df = df[selected_columns].copy()
    #df = moving_median(df, window_size=3)
    IVT_df = IVT_df[~IVT_df["Event"].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
    fix = list(df[df['Eye movement type'] == 'Fixation'].index)

    # compute average direction and openness of both eyes
    df["Gaze direction X"] = df.apply(lambda row: compute_gaze_direction(row["Gaze direction left X"], row["Gaze direction right X"]), axis=1)
    df["Gaze direction Y"] = df.apply(lambda row: compute_gaze_direction(row["Gaze direction left Y"], row["Gaze direction right Y"]), axis=1)
    df["Gaze direction Z"] = df.apply(lambda row: compute_gaze_direction(row["Gaze direction left Z"], row["Gaze direction right Z"]), axis=1)
    df["Eye openness"] = df.apply(lambda row: compute_gaze_direction(row["Eye openness left"], row["Eye openness right"]), axis=1)
    df.loc[df["Eye movement type"] == "EyesNotFound", "Eye openness"] = np.nan
    df = df.iloc[5:].reset_index(drop=True)
    eo_signal = df["Eye openness"]
    #print(np.where(np.isnan(eo_signal)))
    time = df["Recording timestamp"]
    # Parameters (Adjustable)
    WINDOW_LENGTH = 20  # in ms
    VELOCITY_THRESHOLD = 30  # degrees/sec
    MAX_TIME_BETWEEN_FIXATIONS = 75  # ms
    MAX_ANGLE_BETWEEN_FIXATIONS = 0.5  # degrees
    MIN_FIXATION_DURATION = 60  # ms
    df['value_interpolated'] = interpolate_nans(time.values, eo_signal.values)

    #df_new = moving_median(df)
    #df_new = compute_velocity(df_new)
    #df_new = classify_movements(df_new)
    # Detect blinks
    
    print(df.iloc[:30]['value_interpolated'])
    #print(df_new[df_new['Classified Movement'] == 'Saccade'])
    #print(IVT_df[IVT_df["Eye movement type"] == "Saccade"])
    #df_new.to_csv('output.csv', index=False)