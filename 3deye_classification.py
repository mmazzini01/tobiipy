import pandas as pd
import numpy as np

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
            gaze_start = np.array(sample_start[['Gaze point X', 'Gaze point Y']])
            gaze_start = np.append(gaze_start,0)
            eye_center = np.array(df.iloc[center_id][['Eye position X', 'Eye position Y', 'Eye position Z']])
            gaze_end = np.array(sample_end[['Gaze point X', 'Gaze point Y']])
            gaze_end = np.append(gaze_end,0)
            vec_start = eye_center + gaze_start
            vec_end = gaze_end + eye_center
            norm_start = np.linalg.norm(vec_start)
            norm_end   = np.linalg.norm(vec_end)
            vec_start = vec_start / norm_start
            vec_end = vec_end / norm_end
            dot = np.dot(vec_start, vec_end)
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)
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

                # Clamp dot product to avoid numerical errors (ensuring it's between -1 and 1)
                dot_product = np.clip(dot_product, -1.0, 1.0)

                # Compute visual angle in degrees using arccos
                visual_angle = np.degrees(np.arccos(dot_product))

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
    file_path = "raw.xlsx" 
    df = pd.read_excel(file_path)
    IVT_df = pd.read_excel("IVT.xlsx")
    selected_columns = ["Recording timestamp",
                    "Gaze point left X (DACSmm)", "Gaze point left Y (DACSmm)",
                    "Gaze point right X (DACSmm)", "Gaze point right Y (DACSmm)",
                    "Eye position left X (DACSmm)", "Eye position left Y (DACSmm)", "Eye position left Z (DACSmm)",
                    "Eye position right X (DACSmm)", "Eye position right Y (DACSmm)", "Eye position right Z (DACSmm)",
                    "Eye movement type"]
    df= df[~df['Event'].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
    df = df[selected_columns].copy()
    #for the IVT
    IVT_df = IVT_df[~IVT_df['Event'].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
    IVT_df = IVT_df[selected_columns].copy()
    fix = list(df[df['Eye movement type'] == 'Fixation'].index)

    # Apply the function to create average gaze point
    df["Gaze point X"] = df.apply(lambda row: compute_gaze_direction(row["Gaze point left X (DACSmm)"], row["Gaze point right X (DACSmm)"]), axis=1)
    df["Gaze point Y"] = df.apply(lambda row: compute_gaze_direction(row["Gaze point left Y (DACSmm)"], row["Gaze point right Y (DACSmm)"]), axis=1)
    # Apply the function to create average eye position
    df["Eye position X"] = df.apply(lambda row: compute_gaze_direction(row["Eye position left X (DACSmm)"], row["Eye position right X (DACSmm)"]), axis=1)
    df["Eye position Y"] = df.apply(lambda row: compute_gaze_direction(row["Eye position left Y (DACSmm)"], row["Eye position right Y (DACSmm)"]), axis=1)
    df["Eye position Z"] = df.apply(lambda row: compute_gaze_direction(row["Eye position left Z (DACSmm)"], row["Eye position right Z (DACSmm)"]), axis=1)
    # Parameters (Adjustable)
    WINDOW_LENGTH = 20  # in ms
    VELOCITY_THRESHOLD = 30  # degrees/sec
    MAX_TIME_BETWEEN_FIXATIONS = 75  # ms
    MAX_ANGLE_BETWEEN_FIXATIONS = 0.5  # degrees
    MIN_FIXATION_DURATION = 60  # ms
    df = compute_velocity(df)
    df = classify_movements(df)
    sac = np.sum(df['Classified Movement'] == 'Saccade')
    #print(np.where(df[('Classified Movement'== "Saccade")]), np.where(IVT_df[("Eye movement type"]=="Saccade")))
    print(np.where(df['Classified Movement'] == "Saccade"), np.where(IVT_df["Eye movement type"]=="Saccade"))