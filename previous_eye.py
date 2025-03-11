import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
import numpy as np
import pandas as pd
import scipy.signal as signal
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import preprocessing
'''
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

def compute_velocity(df):
    df = df.copy()  # Ensure we don't modify the original dataframe
    df["Velocity"] = np.nan  # Initialize the velocity column
    # Iterate through fixation indexes
    for i in range(1, len(df) - 1):
        # Define the window range
        index_prev = i - 1
        index_post = i + 1
        window = df.iloc[index_prev : index_post].drop(columns=df.columns[-1])

        
        # Check if there are any NaN values in the window
        if window.isna().values.any():
            print(f"Skipping index {i} due to NaN values.")
            continue  # Skip computation if any NaN is found

        # Select preceding and subsequent points

        v1 = df.iloc[index_prev].drop(["Recording timestamp","Velocity"]).to_numpy()
        v2 = df.iloc[index_post].drop(["Recording timestamp","Velocity"]).to_numpy()


        # Compute dot product and norms
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # Avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            continue

        # Compute angle in radians, ensuring numerical stability
        cos_theta = dot_product / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clip to avoid arccos errors
        theta_rad = np.arccos(cos_theta)

        # Convert to degrees
        theta_deg = np.degrees(theta_rad)

        # Compute gaze velocity
        dt = (df.iloc[index_post]['Recording timestamp'] - df.iloc[index_prev]['Recording timestamp']) / 1000
        velocity = theta_deg / dt # Assume timestamps are 1ms apart
        df.at[i, "Velocity"] = velocity  # Store velocity in DataFrame

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

'''
def classify_movements(df1,df2):
    df['Classified Movement'] = df['Eye movement type']  # Preserve existing labels
    filtered_rows = ~pd.isna(df['Velocity']) & (df['Velocity'] > VELOCITY_THRESHOLD)
    df.loc[filtered_rows, 'Classified Movement'] = 'Saccade'
    return df
'''
def classify_movements(df1, df2):
    # Preserva l'etichetta esistente in df1
    df1['Classified Movement'] = df['Eye movement type']
    
    # Crea una maschera che verifica che entrambi i valori di Velocity siano validi e sopra la soglia
    mask = (~pd.isna(df1['Velocity']) & (df1['Velocity'] > VELOCITY_THRESHOLD)) & \
           (~pd.isna(df2['Velocity']) & (df2['Velocity'] > VELOCITY_THRESHOLD))
    
    # Applica la classificazione "Saccade" per le righe che soddisfano la condizione
    df1.loc[mask, 'Classified Movement'] = 'Saccade'
    
    return df1
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


def merge_velocity(df1, df2):
    """
    Combina le colonne 'Velocity' di due DataFrame con la logica:
    - Se entrambi sono NaN → lascia NaN
    - Se uno è NaN e l'altro no → prende il valore non NaN
    - Se entrambi hanno un valore → calcola la media
    
    Args:
    - df1 (pd.DataFrame): Primo DataFrame con colonna 'Velocity'
    - df2 (pd.DataFrame): Secondo DataFrame con colonna 'Velocity'
    
    Returns:
    - pd.Series: Serie con la media dei valori, rispettando le regole sui NaN
    """
    v1 = df1["Velocity"]
    v2 = df2["Velocity"]
    
    # Calcola la media ignorando i NaN, ma mantenendo NaN se entrambi sono NaN
    merged_velocity = np.where(v1.isna() & v2.isna(), np.nan, v1.fillna(0) + v2.fillna(0))
    count_non_nan = (~v1.isna()).astype(int) + (~v2.isna()).astype(int)  # Conta quanti valori validi ci sono
    merged_velocity = np.where(count_non_nan > 0, merged_velocity / count_non_nan, np.nan)  # Calcola la media se possibile
    return pd.Series(merged_velocity, index=df1.index)


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
                        "Eye openness left","Eye openness right","Eye movement type"]
    #df= df[~df['Event'].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
    df = df[selected_columns].copy()
    #df = moving_median(df, window_size=3)
    #IVT_df = IVT_df[~IVT_df["Event"].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
    

    # compute average direction and openness of both eyes
    df = df.iloc[5:].reset_index(drop=True)
    fix = list(df[df['Eye movement type'] == 'Fixation'].index)
    nan_values = list(df.loc[df["Eye movement type"] == "EyesNotFound"].index)
    # Parameters (Adjustable)
    WINDOW_LENGTH = 20  # in ms
    VELOCITY_THRESHOLD = 30  # degrees/sec
    MAX_TIME_BETWEEN_FIXATIONS = 75  # ms
    MAX_ANGLE_BETWEEN_FIXATIONS = 0.5  # degrees
    MIN_FIXATION_DURATION = 60  # ms
    
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
    df_left = df.iloc[:][["Recording timestamp","Gaze direction left X", "Gaze direction left Y", "Gaze direction left Z"]]
    df_right = df.iloc[:][["Recording timestamp","Gaze direction right X", "Gaze direction right Y", "Gaze direction right Z"]]
    df_left_vel = compute_velocity(df_left)
    #print(df_left_vel)
    df_right_vel =  compute_velocity(df_right)
    #print(df)
    '''
    df["Gaze direction X"] = df.apply(lambda row: preprocessing.compute_gaze_direction(row["Gaze direction left X"], row["Gaze direction right X"]), axis=1)
    df["Gaze direction Y"] = df.apply(lambda row: preprocessing.compute_gaze_direction(row["Gaze direction left Y"], row["Gaze direction right Y"]), axis=1)
    df["Gaze direction Z"] = df.apply(lambda row: preprocessing.compute_gaze_direction(row["Gaze direction left Z"], row["Gaze direction right Z"]), axis=1)
    '''
    df_new = classify_movements(df_left_vel, df_right_vel)
    # Detect blinks
   # print(len(df_new))
    dfgt = dfgt.iloc[5:].reset_index(drop=True)
    gt_ind = list(dfgt[dfgt['Eye movement type'] == 'Saccade'].index)
    my_ind = list(df_new[df_new['Classified Movement'] == 'Saccade'].index)
    differenza = list(set(my_ind) - set(gt_ind))
    print("In lista1 e non in lista2:", differenza)
    print(len(my_ind))
    print(len(my_ind))
    #print(gt_ind)
    #print(len(my_ind))
    #print(IVT_df[IVT_df["Eye movement type"] == "Saccade"])
    #df_new.to_csv('output.csv', index=False)