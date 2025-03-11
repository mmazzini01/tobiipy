import pandas as pd
from pygazeanalyser.detectors import ivt

file_path = "csv results/test2_raw.xlsx" 
df = pd.read_excel(file_path)
dfgt = pd.read_excel("csv results/test2ivt_no_noise_red.xlsx")
    #IVT_df = pd.read_excel("csv results/noise_ivt.xlsx")
selected_columns = ["Recording timestamp",
                    "Gaze point X", "Gaze point Y",]
    #df= df[~df['Event'].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
df = df[selected_columns].copy()

df = df.iloc[5:].reset_index(drop=True)
time = df["Recording timestamp"]
# Assuming your DataFrame is named 'df' and has columns 'timestamp', 'x', and 'y'
# Convert your DataFrame to the format expected by PyGaze
raw_data = [(row.timestamp, row.x, row.y) for _, row in df.iterrows()]

# Detect fixations using I-VT (velocity-based) algorithm
# The velocity threshold is in pixels/ms - adjust according to your data characteristics
fixations = detect_fixations(raw_data, 
                            algorithm='velocity', 
                            velocity_threshold=40,  # Adjust this threshold based on your data
                            min_fixation_duration=60)  # Minimum fixation duration in ms

# Convert the results back to a pandas DataFrame for easier analysis
fixation_df = pd.DataFrame(fixations, columns=['start_time', 'end_time', 'duration', 'x', 'y'])