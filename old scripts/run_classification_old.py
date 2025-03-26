import preprocessing
import fixation_detection
import pandas as pd
import settings

if __name__ == '__main__':
    blink_settings = settings.Blink_Settings()
    fix_settings = settings.Fixation_Settings()
    file_path = 'csv results/eye_tracking_recording8.csv'
    gt_file_path = 'csv results/median_blink/test8.xlsx'
    dfgt = pd.read_excel(gt_file_path)
    #dfgt = dfgt[dfgt["Event"].isna()].reset_index(drop=True)
    df = preprocessing.preprocess_data()
    fix = list(df[df['Eye movement type'] == 'Fixation'].index)
    nan_values = list(df[df["Eye movement type"] == "EyesNotFound"].index)
    fd = fixation_detection.FixationDetector(fix_settings,fix,nan_values,blink_settings)
    #bd = blink_detection.BlinkDetector(blink_settings)
    #eo_signal_right = df["Eye openness right"].values
    #eo_signal_left = df["Eye openness left"].values
    #time = df["Recording timestamp"].values
    
    df_out = fd.fixation_detector(df)
    my_ind = list(df_out[df_out['Classified Movement'] == 'Saccade'].index)
    gt_ind = list(dfgt[dfgt['Eye movement type'] == 'Saccade'].index)

    #print(df.shape)
    #print(df_out.loc[[1442,1443]][["Recording timestamp","Classified Movement","Gaze point X","Gaze point Y","Gaze point Z","Gaze direction X","Gaze direction Y","Gaze direction Z"]])
