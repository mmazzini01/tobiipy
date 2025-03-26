import preprocessing
from settings import Fixation_Settings, Blink_Settings
from fixation_detection import FixationDetector
import os


class Tobii_Classifier:
    
    '''Classifier for raw gaze data, mimicking Tobii Pro Lab behavior.
    --------

    This class takes as input the path to a raw gaze data file and automatically processes it to classify
    fixations, saccades, and blinks. It returns two pandas DataFrames:
    - df_full (pd.DataFrame): Full DataFrame with classified eye movements (fixations, saccades, blinks).
    - df_fixation (pd.DataFrame): One row per fixation group, including centroids.


    Example:
    --------
    >>> df_out, fixation_df = Tobii_Classifier("data/subject1.tsv")


    Blink Parameters (Nystrom et al., 2024):
    -----------
    - file_path (str): Path to the raw gaze data file (.csv format).
    - blink_detection (bool): Enable or disable blink detection.
    - Fs (int): Sampling rate of the eye tracker (in Hz).
    - gap_dur (int): Max time gap between missing data to be considered a single blink (in ms).
    - min_amplitude (float): Minimum blink amplitude (as % of full openness).
    - min_separation (int): Minimum duration between two blinks (in ms).
    - debug (bool): Print debug info during blink detection.
    - filter_length (int): Filter length used for blink smoothing (in ms).
    - width_of_blink (int): Width of the blink detection window (in ms).
    - min_blink_dur (int): Minimum duration to classify an event as a blink (in ms).


    Fixation Parameters:
    -----------
    - noise_reduction (str or bool): Type of smoothing applied to gaze data ('median', 'average', or False).
    - noise_reduction_window (int): Window size for gaze smoothing (in samples).
    - velocity_window (int or bool): Window size for velocity smoothing.
    - velocity_threshold (int): Velocity threshold to detect fixations (in degrees/second).
    - label_counter (dict): Internal counter for fixation group labels (used to keep track of grouping).
    - merge_fixations (bool): Whether to merge consecutive fixations if they are close in time.
    - merge_max_time (int): Maximum time allowed between fixations to merge them (in ms).
    - discard_short_fixation (bool): Whether to discard fixations below the duration threshold.
    - min_fixation_duration (int): Minimum allowed fixation duration (in ms).
    '''
    def __new__(cls,
                file_path: str,
                output_path: str,
                blink_detection: bool = True,
                Fs: int = 60,
                gap_dur: int = 40,
                min_amplitude: float = 0.1,
                min_separation: int = 100,
                debug: bool = False,
                filter_length: int = 25,
                width_of_blink: int = 15,
                min_blink_dur: int = 30,
                noise_reduction: str | bool = 'median',
                noise_reduction_window: int = 3,
                velocity_window: int | bool = False,
                velocity_threshold: int = 30,
                label_counter: dict = {'id': 0},
                merge_fixations: bool = False,
                merge_max_time: int = 75,
                discard_short_fixation: bool = True,
                min_fixation_duration: int = 60):
        
        # Create instance normally
        instance = super().__new__(cls)
        # Call __init__ manually
        instance.__init__(
            file_path,
            output_path,
            blink_detection,
            Fs,
            gap_dur,
            min_amplitude,
            min_separation,
            debug,
            filter_length,
            width_of_blink,
            min_blink_dur,
            noise_reduction,
            noise_reduction_window,
            velocity_window,
            velocity_threshold,
            label_counter,
            merge_fixations,
            merge_max_time,
            discard_short_fixation,
            min_fixation_duration
        )
        # Return directly the two DataFrames
        return instance.df_out, instance.fixation_df

    def __init__(
        self,
        file_path: str,
        output_path: str,
        blink_detection: bool = True,
        Fs: int = 60,
        gap_dur: int = 40,
        min_amplitude: float = 0.1,
        min_separation: int = 100,
        debug: bool = False,
        filter_length: int = 25,
        width_of_blink: int = 15,
        min_blink_dur: int = 30,
        noise_reduction: str | bool = 'median',
        noise_reduction_window: int = 3,
        velocity_window: int | bool = False,
        velocity_threshold: int = 30,
        label_counter: dict = {'id': 0},
        merge_fixations: bool = False,
        merge_max_time: int = 75,
        discard_short_fixation: bool = True,
        min_fixation_duration: int = 60,
    ):
        self.file_path = file_path
        self.output_path = output_path

        # Blink detection settings
        self.blink_settings = Blink_Settings(
            blink_detection=blink_detection,
            Fs=Fs,
            gap_dur=gap_dur,
            min_amplitude=min_amplitude,
            min_separation=min_separation,
            debug=debug,
            filter_length=filter_length,
            width_of_blink=width_of_blink,
            min_blink_dur=min_blink_dur
        )

        # Fixation detection settings
        self.fix_settings = Fixation_Settings(
            noise_reduction=noise_reduction,
            noise_reduction_window=noise_reduction_window,
            velocity_window=velocity_window,
            velocity_threshold=velocity_threshold,
            label_counter=label_counter,
            merge_fixations=merge_fixations,
            merge_max_time=merge_max_time,
            discard_short_fixation=discard_short_fixation,
            min_fixation_duration=min_fixation_duration
        )

        # Run classifier immediately and store the result
        self.df_out, self.fixation_df = self.run()

    def data_loader(self):
        return preprocessing.preprocess_data(self.file_path)

    def classifier(self, df):
        fix = list(df[df['Eye movement type'] == 'Fixation'].index)
        nan_values = list(df[df["Eye movement type"] == "EyesNotFound"].index)
        fd = FixationDetector(self.fix_settings, fix, nan_values, self.blink_settings)
        df_out, fixation_df = fd.tobii_classification(df)
        return df_out, fixation_df

    def run(self):
        df = self.data_loader()
        df_full, df_fixation = self.classifier(df)

        # Get base name of input file without extension
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        os.makedirs(self.output_path, exist_ok=True)

        # Create full output file paths
        full_path_full = os.path.join(self.output_path, f"{base_name}_classified.csv")
        full_path_fix = os.path.join(self.output_path, f"{base_name}_fixations.csv")

        # Save CSVs
        df_full.to_csv(full_path_full, index=False)
        df_fixation.to_csv(full_path_fix, index=False)

        print(f"Files saved to:\n → {full_path_full}\n → {full_path_fix}")


if __name__ == '__main__':
    Tobii_Classifier("csv results/eye_tracking_recording8.csv","res_class_out",discard_short_fixation=False,merge_fixations=False)