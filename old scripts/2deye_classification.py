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
    intervals = [samples.loc[i+1,"TIME(2024/10/02 16:49:56.954)"] - samples.loc[i,"TIME(2024/10/02 16:49:56.954)"] for i in range(n - 1)]
    return np.mean(intervals)

def compute_velocity(df):
    global fix, WINDOW_LENGTH
    df['Velocity'] = np.nan  # Initialize velocity column
    # Iterate through fixation indexes
    for pos, center_id in enumerate(fix):
        if pos == 0 or pos == len(df) - 1:
            continue
        else:
            sample_start = df.iloc[center_id]
            sample_end = df.iloc[center_id + 1]
            dt = (sample_end["TIME(2024/10/02 16:49:56.954)"] - sample_start['TIME(2024/10/02 16:49:56.954)'])  # Convert ms to sec
            # angle between two vectors 
            gaze_start = np.array(sample_start[["FPOGX", "FPOGY"]])
            gaze_end = np.array(sample_end[["FPOGX", "FPOGY"]])
            norm_start = np.linalg.norm(gaze_start)
            norm_end   = np.linalg.norm(gaze_end)
            dot = np.dot(gaze_start, gaze_end)
            theta = np.arccos(np.clip((dot/(norm_start*norm_end)),-1,1))
            theta_deg = np.degrees(theta)
            ang_vel = theta_deg / dt
            if ang_vel == 0 or ang_vel == np.nan:
                continue
            else:
                df.at[center_id, 'Velocity'] = ang_vel
    return df

# Function to classify eye movements
def classify_movements(df):
    df['Classified Movement'] = "Saccade" # Preserve existing labels

    for i in range(len(df)):
        if pd.isna(df.iloc[i]['Velocity']):  # If velocity couldn't be computed, keep existing label
            continue

        if df.iloc[i]['Velocity'] < VELOCITY_THRESHOLD:
            df.at[i, 'Classified Movement'] = 'Fixation'

    return df



# Main function to process eye-tracking data
if __name__ == "__main__":
    file_path = "csv results/User 0_all_gaze.csv" 
    df = pd.read_csv(file_path)
    #IVT_df = pd.read_excel("csv results/custom_rawivt.xlsx")
    selected_columns = ["TIME(2024/10/02 16:49:56.954)",
                    "FPOGX", "FPOGY", "FPOGV"]
    df = df[selected_columns].copy()
    fix = list(df[df['FPOGV'] == 1].index)

    WINDOW_LENGTH = 20  # in ms
    VELOCITY_THRESHOLD = 0.35  # degrees/sec
    MAX_TIME_BETWEEN_FIXATIONS = 75  # ms
    MAX_ANGLE_BETWEEN_FIXATIONS = 0.2  # degrees
    MIN_FIXATION_DURATION = 60  # ms
    df = compute_velocity(df)
    df = classify_movements(df)
    sac = np.sum(df['Classified Movement'] == 'Fixation')
    print(sac)