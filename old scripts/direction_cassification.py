import pandas as pd
import numpy as np
from scipy.ndimage import median_filter

def moving_median(df, window_size=3):
    df_filtered = df.copy()
    for col in df.columns:
        if (col != "Eye movement type") and (col != "Recording timestamp"):
            df_filtered[col] = median_filter(df[col], size=window_size, mode='nearest')
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
    avg_interval = compute_average_interval(df, num_samples=100)
    window_sample_count = int(WINDOW_LENGTH / avg_interval) + 1
    if window_sample_count % 2 == 0:
        window_sample_count += 1
    half_window = window_sample_count // 2
    # Iterate through fixation indexes
    for pos, center_id in enumerate(fix):
        if pos == 0 or pos == len(fix) - 1:
            continue  # Skip first and last fixations
        else:
            sample_start = df.iloc[center_id - half_window]
            sample_end = df.iloc[center_id + half_window]
            dt = (sample_end['Recording timestamp'] - sample_start['Recording timestamp']) / 1000  # Convert ms to sec
            direction_start = np.array(sample_start[['Gaze direction X', 'Gaze direction Y', 'Gaze direction Z']])
            direction_end = np.array(sample_end[['Gaze direction X', 'Gaze direction Y', 'Gaze direction Z']])
            dot = np.dot(direction_start, direction_end)
            norm_start = np.linalg.norm(direction_start)
            norm_end   = np.linalg.norm(direction_end)
            theta = np.arccos(dot/(norm_start*norm_end))
            theta_deg = np.degrees(theta)
            ang_vel = theta_deg / dt
            df.at[center_id, 'Velocity'] = ang_vel
    return df

# Function to classify eye movements
def classify_movements(df):
    df['Classified Movement'] = df['Eye movement type']  # Preserve existing labels

    for i in range(len(df)):
        if pd.isna(df.iloc[i]['Velocity']):  # If velocity couldn't be computed, keep existing label
            continue

        if df.iloc[i]['Velocity'] < VELOCITY_THRESHOLD:
            df.at[i, 'Classified Movement'] = 'Fixation'
        else:
            df.at[i, 'Classified Movement'] = 'Saccade'

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
    IVT_df = pd.read_excel("csv results/custom_ivt.xlsx")
    selected_columns = ["Recording timestamp",
                        "Gaze direction left X", "Gaze direction left Y", "Gaze direction left Z",
                        "Gaze direction right X", "Gaze direction right Y", "Gaze direction right Z",
                        "Eye movement type"]
    df= df[~df['Event'].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
    df = df[selected_columns].copy()
    #df = moving_median(df, window_size=3)
    IVT_df = IVT_df[~IVT_df["Event"].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
    fix = list(df[df['Eye movement type'] == 'Fixation'].index)

    # Apply the function to create new columns
    df["Gaze direction X"] = df.apply(lambda row: compute_gaze_direction(row["Gaze direction left X"], row["Gaze direction right X"]), axis=1)
    df["Gaze direction Y"] = df.apply(lambda row: compute_gaze_direction(row["Gaze direction left Y"], row["Gaze direction right Y"]), axis=1)
    df["Gaze direction Z"] = df.apply(lambda row: compute_gaze_direction(row["Gaze direction left Z"], row["Gaze direction right Z"]), axis=1)
    # Parameters (Adjustable)
    WINDOW_LENGTH = 20  # in ms
    VELOCITY_THRESHOLD = 30  # degrees/sec
    MAX_TIME_BETWEEN_FIXATIONS = 75  # ms
    MAX_ANGLE_BETWEEN_FIXATIONS = 0.5  # degrees
    MIN_FIXATION_DURATION = 60  # ms
    df = compute_velocity(df)
    df = classify_movements(df)
    print(df.loc[df["Classified Movement"] == "Saccade"])
    print(IVT_df[IVT_df["Eye movement type"] == "Saccade"])
