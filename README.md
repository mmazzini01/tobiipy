# MasterThesis

# Tobii Eye Tracking Data Classifier

This repository contains a Python implementation for classifying raw Tobii eye tracking data into fixations, saccades, and blinks, mimicking Tobii Pro Lab behavior.

## Features

- Blink detection using eye openness signals
- Fixation detection using I-VT (Velocity Threshold) algorithm
- Saccade identification
- Noise reduction options
- Fixation merging capabilities
- Comprehensive data preprocessing

## Installation


1. Install required packages:


## Usage

You can run the classifier using the command-line interface:

```bash
python run_classification.py <input_file> <output_path> [--discard_short true|false] [--merge_fixations true|false]
```

### Arguments

- `input_file`: Path to the raw eye tracking data CSV file
- `output_path`: Directory where the results will be saved
- `--discard_short`: Whether to discard short fixations (default: true)
- `--merge_fixations`: Whether to merge consecutive fixations (default: true)

### Example

```bash
python run_classification.py input.csv output/ results/ --discard_short true --merge_fixations true
```

### Output

The classifier generates two CSV files:
1. `*_full_classification.csv`: Contains the full classification results
2. `*_fixations.csv`: Contains only the fixation data





