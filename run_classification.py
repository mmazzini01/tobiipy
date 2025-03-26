import argparse
from Tobii_Classifier import Tobii_Classifier

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run Tobii eye tracking data classification')
    
    # Add arguments
    parser.add_argument('input_file', type=str, help='Path to input CSV file containing raw eye tracking data')
    parser.add_argument('output_path', type=str, help='Path to output directory for saving results')
    parser.add_argument('--discard_short', type=str, default='true', choices=['true', 'false'],
                      help='Whether to discard short fixations (default: true)')
    parser.add_argument('--merge_fixations', type=str, default='true', choices=['true', 'false'],
                      help='Whether to merge consecutive fixations (default: true)')

    # Parse arguments
    args = parser.parse_args()
    
    # Convert string boolean arguments to actual booleans
    discard_short = args.discard_short.lower() == 'true'
    merge_fixations = args.merge_fixations.lower() == 'true'

    # Run classification
    Tobii_Classifier(
        file_path=args.input_file,
        output_path=args.output_path,
        discard_short_fixation=discard_short,
        merge_fixations=merge_fixations
    )

if __name__ == '__main__':
    main()