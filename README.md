# Tobii Eye Tracking Data Classifier

A Python-based tool for processing and classifying eye tracking data from Tobii devices, mimicking Tobii Pro Lab behavior. This classifier implements blink detection algorithms and fixation detection using the I-VT (Velocity-Threshold Identification) algorithm.

## Features

- Provides data preprocessing
- Implements Nyström's blink detection algorithm
- Implements I-VT (Velocity-Threshold Identification) fixation detection algorithm
- Outputs detailed CSV files with:
  - Full eye movement classifications with gaze data
  - Gaze data for fixations only

## Requirements Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

You can run the classifier using the command line interface:

```bash
python run_classification.py input_file.csv [options]
```

### Command Line Options

#### Blink Detection Parameters:
- `--blink_detection`: Enable/disable blink detection (default: true)
- `--fs`: Sampling rate in Hz (default: 60)
- `--gap_dur`: Max gap between missing data for blink detection in ms (default: 40)
- `--min_amplitude`: Minimum blink amplitude as % of full openness (default: 0.1)
- `--min_separation`: Minimum time between blinks in ms (default: 100)
- `--debug`: Enable debug output (default: false)
- `--filter_length`: Filter length for blink smoothing in ms (default: 25)
- `--width_of_blink`: Width of blink detection window in ms (default: 15)
- `--min_blink_dur`: Minimum blink duration in ms (default: 30)

#### Fixation Detection Parameters:
- `--noise_reduction`: Type of smoothing ('median', 'average', 'false') (default: median)
- `--noise_reduction_window`: Window size for smoothing in samples (default: 3)
- `--velocity_threshold`: Velocity threshold for fixation detection in degrees/second (default: 30)
- `--merge_fixations`: Enable/disable fixation merging (default: true)
- `--merge_max_time`: Maximum time between fixations to merge in ms (default: 75)
- `--discard_short`: Discard short fixations (default: true)
- `--min_fixation_duration`: Minimum fixation duration in ms (default: 60)

## Output

The classifier creates two CSV files in the `classification_results` directory:
- `*_full_classification.csv`: Complete classification, gaze and duration of all eye tracking data 
- `*_fixations.csv`: gaze data of fixations only

## Example

```bash
python run_classification.py data/raw_eyetracking.csv --velocity_threshold 25 --noise_reduction average
```

## Classes

### Tobii_Classifier
Main class that handles the classification pipeline. It processes raw gaze data to classify fixations, saccades, and blinks.

### BlinkDetector
Implements blink detection algorithms based on eye openness signals.

### FixationDetector
Implements the I-VT algorithm for fixation detection with additional features:
- Noise reduction through median/average filtering
- Classification of eye movements (fixation/saccade)
- Fixation merging based on temporal proximity
- Short fixation filtering
- Eye movement duration

### Settings Classes
- `Blink_Settings`: Configuration for blink detection parameters
- `Fixation_Settings`: Configuration for fixation detection parameters

## References

- Nyström, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation, saccade, and glissade detection in eyetracking data. Behavior Research Methods, 42(1), 188-204.
- Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols. In Proceedings of the 2000 symposium on Eye tracking research & applications (pp. 71-78).





