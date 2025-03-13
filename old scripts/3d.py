import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
import numpy as np
import pandas as pd
import scipy.signal as signal
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import preprocessing

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
        if not window[['Gaze direction X', 'Gaze direction Y', 'Gaze direction Z']].isna().values.any():
            for col in df.columns:
                if (col != "Eye movement type") and (col != "Recording timestamp"):
                    median = np.median([df.iloc[prev][col],df.iloc[id][col],df.iloc[post][col]])
                    df_filtered.at[id,col] = median
    return df_filtered
'''
def moving_median(arr):
    arr = np.asarray(arr)  # Assicura che l'input sia un array NumPy
    result = np.full_like(arr, np.nan)  # Array di output inizializzato a NaN
    
    for i in range(len(arr)):
        if i == 0 or i == len(fix)-1:
            continue
        prev = i-1
        post = i+1
        window = arr[prev:post]
        if not np.isnan(window).any():
             result[i] = np.median(window)
        else:
            result[i] = arr[i]
    return result
'''

def compute_velocity(df):
    global fix, nan_values
    df['Velocity'] = np.nan  # Inizializza la colonna Velocity
    # Itera sugli indici delle fixazioni
    for i in range(len(fix)):
        if i == 0:
            continue  # Salta la prima fissazione
        else:
            index_fix = fix[i]
            index_prev = index_fix - 1
            if index_prev not in nan_values:
                sample_start = df.iloc[index_prev]
                sample_end = df.iloc[index_fix]
                dt = (sample_end['Recording timestamp'] - sample_start['Recording timestamp']) / 1000  # Converti ms in sec

                # Estrai la direzione dello sguardo e la posizione dell'occhio per il sample precedente
                direction_start = np.array(sample_start[['Gaze direction X', 'Gaze direction Y', 'Gaze direction Z']])
                eye_pos_start = np.array(sample_start[['Eye position X', 'Eye position Y', 'Eye position Z']])
                
                # Estrai la direzione dello sguardo e la posizione dell'occhio per il sample corrente
                direction_end = np.array(sample_end[['Gaze direction X', 'Gaze direction Y', 'Gaze direction Z']])
                eye_pos_end = np.array(sample_end[['Eye position X', 'Eye position Y', 'Eye position Z']])
                
                # Calcola il punto verso cui era diretto lo sguardo nel sample precedente
                P_prev = eye_pos_start + direction_start
                # Trasla il vettore precedente affinché parta dalla posizione attuale dell'occhio
                translated_direction_prev = P_prev - eye_pos_end

                # Normalizza i vettori
                norm_translated_prev = np.linalg.norm(translated_direction_prev)
                if norm_translated_prev == 0:
                    theta_deg = np.nan
                else:
                    translated_direction_prev_norm = translated_direction_prev / norm_translated_prev

                    # direction_end è già un vettore unitario
                    # Calcola l'angolo tramite il prodotto scalare
                    dot = np.dot(translated_direction_prev_norm, direction_end)
                    theta = np.arccos(np.clip(dot, -1, 1))
                    theta_deg = np.degrees(theta)
                
                # Calcola la velocità angolare (°/sec)
                ang_vel = theta_deg / dt if dt != 0 else np.nan
                df.at[index_fix, 'Velocity'] = ang_vel
    return df


# Function to classify eye movements
'''
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
'''
def classify_movements(df):
    df['Classified Movement'] = df['Eye movement type']  # Preserve existing labels
    filtered_rows = ~pd.isna(df['Velocity']) & (df['Velocity'] > VELOCITY_THRESHOLD)
    df.loc[filtered_rows, 'Classified Movement'] = 'Saccade'
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
    file_path = "csv results/test2_raw.xlsx" 
    df = pd.read_excel(file_path)
    dfgt = pd.read_excel("csv results/test2_ivt_noblink.xlsx")
    #IVT_df = pd.read_excel("csv results/noise_ivt.xlsx")
    selected_columns = ["Recording timestamp",
                        "Gaze direction left X", "Gaze direction left Y", "Gaze direction left Z",
                        "Gaze direction right X", "Gaze direction right Y", "Gaze direction right Z",
                        "Eye position left X (DACSmm)", "Eye position left Y (DACSmm)", "Eye position left Z (DACSmm)",
                        "Eye position right X (DACSmm)", "Eye position right Y (DACSmm)", "Eye position right Z (DACSmm)",
                        "Eye openness left","Eye openness right","Eye movement type"]
    #df= df[~df['Event'].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
    df = df[selected_columns]
    #df = moving_median(df, window_size=3)
    #IVT_df = IVT_df[~IVT_df["Event"].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
    

    # compute average direction and openness of both eyes
    df = df.iloc[5:].reset_index(drop=True)
    fix = list(df[df['Eye movement type'] == 'Fixation'].index)
    nan_values = list(df.loc[df["Eye movement type"] == "EyesNotFound"].index)
    time = df["Recording timestamp"]
    # Parameters (Adjustable)
    WINDOW_LENGTH = 20  # in ms
    VELOCITY_THRESHOLD = 30  # degrees/sec
    MAX_TIME_BETWEEN_FIXATIONS = 75  # ms
    MAX_ANGLE_BETWEEN_FIXATIONS = 0.5  # degrees
    MIN_FIXATION_DURATION = 60  # ms
    '''
    fil_left_x= moving_median(df["Gaze direction left X"])
    fil_left_y= moving_median(df["Gaze direction left Y"])
    fil_left_z= moving_median(df["Gaze direction left Z"])
    fil_right_x= moving_median(df["Gaze direction right X"])        
    fil_right_y= moving_median(df["Gaze direction right Y"])
    fil_right_z= moving_median(df["Gaze direction right Z"])
    df['Gaze direction left X'] = fil_left_x
    df['Gaze direction left Y'] = fil_left_y
    df['Gaze direction left Z'] = fil_left_z
    df['Gaze direction right X'] = fil_right_x
    df['Gaze direction right Y'] = fil_right_y
    df['Gaze direction right Z'] = fil_right_z
    '''
    df["Gaze direction X"] = df.apply(lambda row: preprocessing.compute_gaze_direction(row["Gaze direction left X"], row["Gaze direction right X"]), axis=1)
    df["Gaze direction Y"] = df.apply(lambda row: preprocessing.compute_gaze_direction(row["Gaze direction left Y"], row["Gaze direction right Y"]), axis=1)
    df["Gaze direction Z"] = df.apply(lambda row: preprocessing.compute_gaze_direction(row["Gaze direction left Z"], row["Gaze direction right Z"]), axis=1)
    df["Eye position X"] = df.apply(lambda row: preprocessing.compute_gaze_direction(row["Eye position left X (DACSmm)"], row["Eye position right X (DACSmm)"]), axis=1)
    df["Eye position Y"] = df.apply(lambda row: preprocessing.compute_gaze_direction(row["Eye position left Y (DACSmm)"], row["Eye position right Y (DACSmm)"]), axis=1)    
    df["Eye position Z"] = df.apply(lambda row: preprocessing.compute_gaze_direction(row["Eye position left Z (DACSmm)"], row["Eye position right Z (DACSmm)"]), axis=1)
    df_new = moving_median(df)
    df_new = compute_velocity(df_new)
    df_new = classify_movements(df_new)
    # Detect blinks
   # print(len(df_new))
    dfgt = dfgt.iloc[5:].reset_index(drop=True)
    gt_ind = list(dfgt[dfgt['Eye movement type'] == 'Saccade'].index)
    my_ind = list(df_new[df_new['Classified Movement'] == 'Saccade'].index)
    #print(df_new.iloc[[34,35,36,106,107,108,417,418,419]][['Gaze direction X', 'Gaze direction Y', 'Gaze direction Z',"Velocity"]])
    print(len(my_ind))
    print(my_ind)
    #print(IVT_df[IVT_df["Eye movement type"] == "Saccade"])
    #df_new.to_csv('output.csv', index=False)