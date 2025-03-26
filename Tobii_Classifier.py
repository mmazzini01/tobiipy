import os
import preprocessing
from settings import Fixation_Settings, Blink_Settings
from fixation_detection import FixationDetector
import sys


class Tobii_Classifier:
    '''
    Classifier for raw gaze data, mimicking Tobii Pro Lab behavior.
    --------

    This class takes as input the path to a raw gaze data file and automatically processes it to classify
    fixations, saccades, and blinks. It saves the results as CSV files in the specified output directory.

    Example:
    --------
    >>> Tobii_Classifier("data/raw.csv", output_path="results/")

    Blink Parameters (Nystrom et al., 2024):
    -----------
    - file_path: Path to the raw gaze data file (.csv format).
    - blink_detection: Enable or disable blink detection.
    - Fs: Sampling rate of the eye tracker (in Hz).
    - gap_dur: Max time gap between missing data to be considered a single blink (in ms).
    - min_amplitude: Minimum blink amplitude (as % of full openness).
    - min_separation: Minimum duration between two blinks (in ms).
    - debug: Print debug info during blink detection.
    - filter_length: Filter length used for blink smoothing (in ms).
    - width_of_blink: Width of the blink detection window (in ms).
    - min_blink_dur: Minimum duration to classify an event as a blink (in ms).

    Fixation Parameters (I-VT):
    -----------
    - noise_reduction: Type of smoothing applied to gaze data ('median', 'average', or False).
    - noise_reduction_window: Window size for gaze smoothing (in samples).
    - velocity_window: Window size for velocity smoothing.
    - velocity_threshold: Velocity threshold to detect fixations (in degrees/second).
    - label_counter: Internal counter for fixation group labels (used to keep track of grouping).
    - merge_fixations: Whether to merge consecutive fixations if they are close in time.
    - merge_max_angle: Maximum angle allowed between fixations to merge them (in degrees).
    - merge_max_time: Maximum time allowed between fixations to merge them (in ms).
    - discard_short_fixation: Whether to discard fixations below the duration threshold.
    - min_fixation_duration: Minimum allowed fixation duration (in ms).
    '''

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

        # Run the full pipeline immediately
        self.run()

    def data_loader(self):
        return preprocessing.preprocess_data(self.file_path)

    def classifier(self, df):
        fix = list(df[df['eye_movement_type'] == 'fixation'].index)
        nan_values = list(df[df["eye_movement_type"] == "eyes_not_found"].index)
        fd = FixationDetector(self.fix_settings, fix, nan_values, self.blink_settings)
        df_out, fixation_df = fd.tobii_classification(df)
        return df_out, fixation_df

    def run(self):
        df = self.data_loader()
        df_full, df_fixation = self.classifier(df)

        # Extract base name and ensure output folder exists
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        os.makedirs(self.output_path, exist_ok=True)

        # Output file paths
        full_path_full = os.path.join(self.output_path, f"{base_name}_full_classification.csv")
        full_path_fix = os.path.join(self.output_path, f"{base_name}_fixations.csv")

        # Save the files
        df_full.to_csv(full_path_full, index=False)
        df_fixation.to_csv(full_path_fix, index=False)

        print(f"Files saved to:\n -> {full_path_full}\n -> {full_path_fix}")

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    discard_fix = sys.argv[3].lower() == 'true'
    merge = sys.argv[3].lower() == 'true'
    Tobii_Classifier(file_path = input_file,
                     output_path = output_file,
                     discard_short_fixation=discard_fix,merge_fixations=merge)
