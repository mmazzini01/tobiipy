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
        self.velocity_window = True
        self.velocity_threshold = 30
        self.label_counter = {'id':0}
        self.merge_fixations = True
        self.merge_max_time = 75
        self.merge_max_angle = 0.5
        self.discard_short_fixations = True
        self.min_fixation_duration = 60
if __name__ == '__main__':
    #blink_settings = Blink_Settings()
    fix_settings = Fixation_Settings()
    file_path = 'csv results/output.csv'
    gt_file_path = 'csv results/test13_median.xlsx'
    dfgt = pd.read_excel(gt_file_path)
    df = preprocessing.filter_data(file_path)
    df.at[217,"Eye movement type"] = 'Unclassified'
    fix = list(df[df['Eye movement type'] == 'Fixation'].index)
    nan_values = list(df[df["Eye movement type"] == "EyesNotFound"].index)
    fd = fixation_detection.FixationDetector(fix_settings,fix,nan_values)
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
    FP = list(set(my_ind) - set(gt_ind))
    FN = list(set(gt_ind) - set(my_ind))
    print(df_out[(df_out["Velocity"] >= 29) & (df_out["Velocity"] <= 31)])
    print(df_out.loc[:][["Recording timestamp","Eye movement type","Classified Movement","Group","Velocity","Eye movement duration"]])
    print(fix_settings.label_counter)

    print("False Postive:", FP)
    print("False Negative:", FN)
