from dataclasses import dataclass, field

@dataclass
class Blink_Settings:
    '''Settings related to blink detection.

    Attributes:
        blink_detection: Enable or disable blink detection.
        Fs: Sample rate of eye tracker (Hz).
        gap_dur: Max gaps between periods of data loss (ms).
        min_amplitude: % of fully open eye (0.1 - 10%).
        min_separation: Minimum time (ms) between blinks.
        debug: Enable debugging output.
        filter_length: Filter length (ms).
        width_of_blink: Blink detection window (ms).
        min_blink_duration: Minimum blink duration (ms).
    '''
    blink_detection: bool = field(default=True)
    Fs: int = field(default=60)
    gap_dur: int = field(default=40)
    min_amplitude: float = field(default=0.1)
    min_separation: int = field(default=100)
    debug: bool = field(default=False)
    filter_length: int = field(default=25)
    width_of_blink: int = field(default=15)
    min_blink_dur: int = field(default=30)




from dataclasses import dataclass, field

@dataclass
class Fixation_Settings:
    '''Settings related to fixation detection.

    Attributes:
        noise_reduction: Method used for noise reduction: None/False, 'median', or 'average'.
        noise_reduction_window: Size of the window used for noise reduction.
        velocity_window: Window size for velocity calculation (None/False if not used).
        velocity_threshold: Threshold for velocity-based fixation detection (degrees/sec).
        label_counter: Dictionary to store group index (DO NOT CHANGE).
        merge_fixations: Whether to merge consecutive fixations (None/False for default behavior).
        merge_max_time: Maximum time (ms) between consecutive fixations to allow merging.
        discard_short_fixation: Wheter to reclassify short fixations as "Unclassified"
        min_fixation_duration: Minimun duration of a fixation (ms)

    '''
    noise_reduction: str | None | bool = field(default='median')
    noise_reduction_window: int = field(default=3)
    velocity_window: int | None | bool = field(default=None)
    velocity_threshold: int = field(default=30)
    label_counter: dict[str, int] = field(default_factory=lambda: {'id': 0})
    merge_fixations: bool | None = field(default=None)
    merge_max_angle: float = field(default= 0.5)
    merge_max_time: int = field(default=75)
    discard_short_fixation: bool | None = field(default=True)
    min_fixation_duration: int = field(default=60)
