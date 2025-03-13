import numpy as np
import preprocessing
import blink_detection
import fixation_detection
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter, find_peaks

class Blink_Settings():
    def __init__(self):
        self.blink_detection = True
        self.Fs = 60  # Sample rate of eye tracker
        self.gap_dur = 40  # Max gaps between period of data loss, interpolate smaller gaps
        self.min_amplitude = 0.1  # % of fully open eye (0.1 - 10%)
        self.min_separation = 100  # Min separation between blinks
        self.debug = False

        self.filter_length = 25  # in ms
        self.width_of_blink = 15  # in ms width of peak to initially detect
        self.min_blink_dur = 30  # Reject blinks shorter than 30 ms
    
class Fixation_Settings():
    def __init__(self):
        self.eye_selection = 'average'
        self.noise_reduction = 'median'
        self.noise_reduction_window = 3
        self.velocity_window = None
        self.velocity_threshold = 30
        self.label_counter = {'id':0} # DO NOT CHANGE USED FOR GROUP INDEX 
        self.merge_fixations = None
        self.merge_max_time = 75
        self.merge_max_angle = 0.5
        self.discard_short_fixations = None
        self.min_fixation_duration = 60
if __name__ == '__main__':
    blink_settings = Blink_Settings()
    fix_settings = Fixation_Settings()
    file_path = 'csv results/output8.csv'
    gt_file_path = 'csv results/median_blink/test8.xlsx'
    dfgt = pd.read_excel(gt_file_path)
    #dfgt = dfgt[dfgt["Event"].isna()].reset_index(drop=True)
    df = preprocessing.filter_data(file_path)
    fix = list(df[df['Eye movement type'] == 'Fixation'].index)
    nan_values = list(df[df["Eye movement type"] == "EyesNotFound"].index)
    fd = fixation_detection.FixationDetector(fix_settings,fix,nan_values,blink_settings)
    #bd = blink_detection.BlinkDetector(blink_settings)
    #eo_signal_right = df["Eye openness right"].values
    #eo_signal_left = df["Eye openness left"].values
    #time = df["Recording timestamp"].values
    '''
    df_blink, eye_openness_signal_vel = bd.blink_detector_eo(time, eo_signal_left,eo_signal_right , blink_settings.Fs, filter_length=blink_settings.filter_length,
                                                             gap_dur=blink_settings.gap_dur,
                                                             width_of_blink=blink_settings.width_of_blink,
                                                             min_separation=blink_settings.min_separation)'
    '''
    
    df_out = fd.fixation_detector(df)
    my_ind = list(df_out[df_out['Classified Movement'] == 'Saccade'].index)
    gt_ind = list(dfgt[dfgt['Eye movement type'] == 'Saccade'].index)

    print(df.shape)
    #print(df_out.loc[[]][["Recording timestamp","Eye movement type","Classified Movement","Group","Velocity","Eye movement duration"]])

    with open("comparison.txt", "w") as f:
        FP = 0  # False Positives
        FN = 0  # False Negatives
        f.write("ID  MINE      GT       VEL   DUR    GROUP_ID")
        f.write("\n")
        for i in range(len(df_out)):
            classified_movement = df_out.loc[i, "Classified Movement"]
            eye_movement_type = dfgt.loc[i, "Eye movement type"]
            velocity = df_out.loc[i,"Velocity"].round(1)
            eye_mov_tipe_index = df_out.loc[i,"Label Group"]
            duration = df_out.loc[i,"Eye movement duration"].round(0)
            marker = "            <----------------------------" if classified_movement != eye_movement_type else ""
            f.write(f"{i}  {classified_movement}  {eye_movement_type}  {velocity}  {duration}  {eye_mov_tipe_index}{marker}\n")
            if classified_movement == "Fixation" and eye_movement_type == "Saccade":
                FN += 1  #
            if classified_movement == "Saccade" and eye_movement_type == "Fixation":
                FP += 1  
        
        f.write("\n")  
        f.write(f"False Positive: {FP}\n")
        f.write(f"False Negative: {FN}\n")
