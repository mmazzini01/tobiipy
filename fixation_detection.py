import pandas as pd
import numpy as np
import blink_detection

class FixationDetector():
    """Class for detecting and analyzing fixations in eye tracking data.
    
    Handles noise reduction, fixation detection, classification of eye movements,
    and computation of fixation metrics.
    """
    
    def __init__(self, settings, fix, nan_values, blink_settings):
        """Initialize the FixationDetector.
        
        Args:
            settings: Configuration settings for fixation detection
            fix: List of fixation indices
            nan_values: List of indices containing NaN values
            blink_settings: Settings for blink detection
        """
        self.settings = settings
        self.fix = fix
        self.nan_values = nan_values
        self.blink_detection = blink_detection.BlinkDetector(blink_settings)

    def noise_reduction(self, df):
        """Apply noise reduction to gaze data using median or average filtering.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing gaze data
            
        Returns:
            pd.DataFrame: DataFrame with noise reduction applied
            
        Raises:
            ValueError: If noise reduction window size is not a positive odd number
        """
        # If noise_reduction is None or False, return df unchanged
        if not self.settings.noise_reduction:
            return df
            
        df_filtered = df.copy()
        
        # Validate window size
        if self.settings.noise_reduction_window <= 0 or self.settings.noise_reduction_window % 2 == 0:
            raise ValueError("Noise reduction window must be a positive odd number.")
            
        window_size = self.settings.noise_reduction_window // 2
        
        # Apply filtering for each fixation point
        for i in range(1, len(self.fix) - 1):  # Avoid first and last indices
            id = self.fix[i]
            prev = self.fix[i - window_size]
            post = self.fix[i + window_size]
            window = df.loc[prev:post]
            
            # Only filter if no NaN values in gaze coordinates
            if not window[['gaze_point_x', 'gaze_point_y', 'gaze_point_z']].isna().values.any():
                for col in df.columns:
                    # Skip certain columns that shouldn't be filtered
                    if col not in ["eye_movement_type","date_time", "recording_timestamp","eye_openness_left","eye_openness_right",
                                    "left_gaze_origin_validity", "right_gaze_origin_validity", "left_pupil_diameter",
                                    "right_pupil_diameter"]:
                        if self.settings.noise_reduction == 'median':
                            value = np.median([df.at[prev, col], df.at[id, col], df.at[post, col]])
                        elif self.settings.noise_reduction == 'average':
                            value = np.mean([df.at[prev, col], df.at[id, col], df.at[post, col]])
                        df_filtered.at[id, col] = value
        return df_filtered
    
    def compute_angle(self, array1, array2, matrix=None):
        """Compute angle between two vectors or arrays of vectors.
        
        Args:
            array1: First vector or array of vectors
            array2: Second vector or array of vectors  
            matrix (bool): If True, compute angles for arrays of vectors
            
        Returns:
            float or np.array: Angle(s) in degrees between vectors
        """
        if not matrix:   
            dot = np.dot(array1, array2)
            theta = np.arccos((np.clip(dot,-1,1)))
            theta_deg = np.degrees(theta)
            return theta_deg
        else:
            dot = np.sum(array1 * array2, axis=1)  
            norm_start = np.linalg.norm(array1, axis=1)  
            norm_end   = np.linalg.norm(array2, axis=1)  
            theta = np.arccos(np.clip(dot / (norm_start * norm_end), -1, 1)) 
            theta_deg = np.degrees(theta)  
            return theta_deg 
    
    def compute_velocity(self, df):
        """Compute angular velocity between consecutive gaze points.
        
        Args:
            df (pd.DataFrame): DataFrame containing gaze data
            
        Returns:
            pd.DataFrame: DataFrame with velocity column added
            
        Raises:
            ValueError: If sampling rate is not 60 Hz
        """
        df['velocity'] = np.nan
        
        # Only implemented for 60 Hz
        if self.blink_detection.settings.Fs == 60:
            fix_indices = np.array(self.fix[1:])  # skip the first point
            prev_indices = fix_indices - 1

            # Filter out invalid previous indices (e.g., from nan_values)
            valid_mask = ~np.isin(prev_indices, self.nan_values)
            fix_indices = fix_indices[valid_mask]
            prev_indices = prev_indices[valid_mask]

            # Get timestamps and direction vectors
            t_curr = df.loc[fix_indices, 'recording_timestamp'].values
            t_prev = df.loc[prev_indices, 'recording_timestamp'].values
            dt = (t_curr - t_prev) / 1000  # convert ms to sec

            dirs_prev = df.loc[prev_indices, ['gaze_direction_x', 'gaze_direction_y', 'gaze_direction_z']].astype(float).values
            dirs_curr = df.loc[fix_indices, ['gaze_direction_x', 'gaze_direction_y', 'gaze_direction_z']].astype(float).values

            # Normalize direction vectors
            v1 = dirs_prev / np.linalg.norm(dirs_prev, axis=1, keepdims=True)
            v2 = dirs_curr / np.linalg.norm(dirs_curr, axis=1, keepdims=True)

            # Compute angle in degrees
            dot_products = np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)
            angles_rad = np.arccos(dot_products)
            angles_deg = np.degrees(angles_rad)

            # Angular velocity
            angular_velocity = angles_deg / dt

            # Assign results back to the DataFrame
            df.loc[fix_indices, 'velocity'] = angular_velocity
            return df
        else:
            raise ValueError("Sample rate different from baseline. This velocity calculation has been implemented for 60 HZ sample rate")
        
    
    def classify_movement(self, df):
        """Classify eye movements as fixations or saccades based on velocity.
        
        Args:
            df (pd.DataFrame): DataFrame with velocity data
            
        Returns:
            pd.DataFrame: DataFrame with classified movements
        """
        df['classified_movement'] = df['eye_movement_type']  # Preserve existing labels
        filtered_rows = ~pd.isna(df['velocity']) & (df['velocity'] >= self.settings.velocity_threshold)
        df.loc[filtered_rows, 'classified_movement'] = 'saccade'
        
        # If sample is after a nan, take the label of the next sample
        df['classified_movement'] = df['classified_movement'].where(~(pd.isna(df['velocity']) & (df['classified_movement'] == 'fixation')
                                                                      ), df['classified_movement'].shift(-1))
        df.at[0,"classified_movement"] = "fixation"
        df["diff"] = df["velocity"].diff()
        return df

    def blink_call(self, df):
            """Detect blinks in eye tracking data.
            
            Args:
                df (pd.DataFrame): DataFrame with eye openness data
                
            Returns:
                pd.DataFrame: DataFrame with blinks detected and labeled
            """
            blink_class = self.blink_detection
            if blink_class.settings.blink_detection:
                t =df["recording_timestamp"].values
                eye_op_left = df["eye_openness_left"].values
                eye_op_right = df["eye_openness_right"].values
                blink_df = blink_class.blink_detector_eo(t,eye_op_left,eye_op_right)
                gap = t
                blink_starts = blink_df['onset'].values[:, None]  
                blink_ends = blink_df['offset'].values[:, None]   
                # For each timestamp, check if it's in any blink interval
                mask = ((gap >= blink_starts) & (gap < blink_ends)).any(axis=0)

                df.loc[mask, 'classified_movement'] = 'blink'
                
                return df
            else:
                return df
    
    def assign_unique_group_number(self, df):
        """Assign unique group numbers to consecutive segments with same movement type.
        
        Args:
            df (pd.DataFrame): DataFrame with classified movements
            
        Returns:
            pd.DataFrame: DataFrame with group numbers assigned
        """
        #Assign group numbers
        df['group'] = (df['classified_movement'] != df['classified_movement'].shift()).cumsum()
        return df
    
    def merge_adjacent_fixations(self, df):
        """Merge adjacent fixations based on time gap and angular velocity thresholds.
        
        Args:
            df (pd.DataFrame): DataFrame with classified fixations
            
        Returns:
            pd.DataFrame: DataFrame with merged fixations
        """
        df = self.assign_unique_group_number(df)
        if self.settings.merge_fixations:
            df = df.copy()
            df_fix = df[df['classified_movement'] == "fixation"].copy()

            # Compute time differences between consecutive fixation groups
            group_last_timestamps = df_fix.groupby("group")['recording_timestamp'].last()
            group_first_timestamps = df_fix.groupby("group")['recording_timestamp'].first()
            time_gaps = group_first_timestamps.shift(-1) - group_last_timestamps  # Time gap to next group

            # Compute average eye position between consecutive fixations
            eye_last = df_fix.groupby("group")[["eye_position_x","eye_position_y","eye_position_z"]].last().to_numpy()
            eye_first = df_fix.groupby("group")[["eye_position_x","eye_position_y","eye_position_z"]].first().shift(-1).to_numpy()
            
            avg_eye= np.mean([eye_first,eye_last], axis=0)

            # Compute visual angles between consecutive fixations
            gaze_last = df_fix.groupby("group")[["gaze_point_x","gaze_point_y","gaze_point_z"]].last().to_numpy()
            gaze_first = df_fix.groupby("group")[["gaze_point_x","gaze_point_y","gaze_point_z"]].first().shift(-1).to_numpy()

            dir_to_last = gaze_last - avg_eye
            dir_to_first = gaze_first -avg_eye
            angles = self.compute_angle(dir_to_first,dir_to_last, matrix=True)

            # Determine which fixations should be merged
            merge_groups = (time_gaps < self.settings.merge_max_time) & (angles < self.settings.merge_max_angle)

            # Map merge decisions back to the original DataFrame
            merge_flags = df_fix['group'].map(merge_groups.fillna(False))

            df["merged"] = df["group"].map(merge_flags.fillna(False))
            pd.set_option('future.no_silent_downcasting', True)
            df["merged"] = df["merged"].fillna(False).astype(bool)


            df.loc[df["merged"], "classified_movement"] = "fixation"
            # assign new group numbers
            df.drop(columns=['group'], inplace=True)
            df = self.assign_unique_group_number(df)
            # Drop 'Merged' column (optional)
            #df.drop(columns=['merged'], inplace=True)
            return df
        else:
            return df
    
    def compute_eye_movement_duration(self, df):
        """Compute duration of each eye movement segment.
        
        Args:
            df (pd.DataFrame): DataFrame with grouped movements
            
        Returns:
            pd.DataFrame: DataFrame with duration computed for each movement
        """
        df = df.copy()
        sample_rate_ms = int((1/self.blink_detection.settings.Fs) * 1000)
    
        first_timestamps = df.groupby('group')['recording_timestamp'].first()
        last_timestamps = df.groupby('group')['recording_timestamp'].last()

        # Step 3: Compute the start time (midpoint of previous last and current first)
        prev_last_timestamps = last_timestamps.shift(1)  # Last timestamp of previous group
        start_times = ((prev_last_timestamps + first_timestamps) // 2).round(0)

        # Step 4: Compute the end time (midpoint of current last and next first)
        next_first_timestamps = first_timestamps.shift(-1)  # First timestamp of next group
        end_times = ((last_timestamps + next_first_timestamps) // 2).round(0)

        end_times.iloc[-1] = last_timestamps.iloc[-1]  # Last group ends at its last timestamp
        start_times.iloc[-1] = last_timestamps.iloc[-2]
        # First group duration = (First timestamp of next group - First timestamp of first group)
        start_times.iloc[0] = 0

        # Step 6: Compute the final duration
        durations = end_times - start_times

        df['is_blink'] = df['classified_movement'] == 'blink'
        blink_groups = df[df['is_blink']]['group'].unique()

        # Create a boolean mask for groups that have Blink before or after
        group_series = pd.Series(durations.index, index=durations.index)
        has_blink_before = group_series.shift(1).isin(blink_groups)
        has_blink_after = group_series.shift(-1).isin(blink_groups)

        # Adjust durations based on blink presence
        durations[has_blink_before] -= sample_rate_ms // 2
        durations[has_blink_after] += sample_rate_ms // 2

        # Step 7: Map the computed durations back to the original DataFrame
        df['duration'] = df['group'].map(durations)
        return df
    
    def assign_movement_index(self, row):
        """Assign incremental index to each movement group.
        
        Args:
            row (pd.Series): DataFrame row containing movement data
            
        Returns:
            int: Movement index for the group
        """
        label = row['classified_movement']  # Use the correct column name
        group = row['group']

        if group > self.settings.label_counter.get('id'):
            # Increment label-specific counter
            self.settings.label_counter[label] = self.settings.label_counter.get(label, 0) + 1
            self.settings.label_counter['id'] +=1

        return self.settings.label_counter[label]

    def discard_short_fixation(self, df):
        """Remove fixations shorter than minimum duration threshold.
        
        Args:
            df (pd.DataFrame): DataFrame with fixation durations
            
        Returns:
            pd.DataFrame: DataFrame with short fixations removed
        """
        if self.settings.discard_short_fixation:
            df.loc[(df["classified_movement"] == "fixation") & (df["duration"] < self.settings.min_fixation_duration), "classified_movement"] = "unclassified"
            # assign new group numbers
            df.drop(columns=['group'], inplace=True)
            df = self.assign_unique_group_number(df)
            # recompute duration
            df = self.compute_eye_movement_duration(df)
            # compute group index
            df['group_index'] = df.apply(self.assign_movement_index, axis=1)
            return df
        else:
            df['group_index'] = df.apply(self.assign_movement_index, axis=1)
            return df
        
        
    def compute_fixation_centroids(self, df):
        """Calculate centroids for each fixation group.
        
        Args:
            df (pd.DataFrame): DataFrame with fixation data
            
        Returns:
            pd.DataFrame: DataFrame with centroid coordinates added
        """
        # Filter only rows where 'Classified Movement' is 'Fixation'
        fixation_df = df[df['classified_movement'] == 'fixation']

        # Compute the mean for each group (X, Y, left & right pupil diameter)
        centroids = fixation_df.groupby('group')[
            ['point_on_display_x', 'point_on_display_y', 'left_pupil_diameter', 'right_pupil_diameter']
        ].mean()

        # Rename the columns
        centroids.columns = ['x', 'y', 'pupil_left', 'pupil_right']
        centroids.reset_index(inplace=True)

        # Merge the centroids back to the original DataFrame
        df = df.merge(centroids, on='group', how='left')

        # Set centroid columns to NaN for rows that are not 'Fixation'
        mask = df['classified_movement'] != 'fixation'
        df.loc[mask, ['x', 'y', 'pupil_left', 'pupil_right']] = None

        return df

    
    def compute_fixation_df(self, df):
        """Create summary DataFrame with one row per fixation.
        
        Args:
            df (pd.DataFrame): Full eye tracking DataFrame
            
        Returns:
            pd.DataFrame: Summary DataFrame with fixation metrics
        """
        # Filter only 'Fixation' movements
        fixation_df = df[df['classified_movement'] == 'fixation'].copy()

        # Select only one row per 'Label Group' (e.g., the first occurrence)
        fixation_df = fixation_df.drop_duplicates(subset='group_index')

        # Keep only the desired columns
        columns = [
            "classified_movement", "recording_timestamp",'x', 'y', 'pupil_left', 'pupil_right',
            "duration", "group_index"
        ]
        fixation_df = fixation_df[columns]

        return fixation_df

    
    def tobii_classification(self, df):
        """Main pipeline for classifying  eye tracking data as Tobii pro lab.
        
        Applies full processing pipeline:
        - Noise reduction
        - Velocity computation
        - Movement classification
        - Blink detection
        - Fixation merging
        - Duration computation
        - Short fixation removal
        - Centroid computation
        
        Args:
            df (pd.DataFrame): Raw eye tracking DataFrame
            
        Returns:
            tuple: (processed DataFrame, fixation summary DataFrame)
        """
        df = self.noise_reduction(df)
        df = self.compute_velocity(df)
        df = self.classify_movement(df)
        df = self.blink_call(df)
        df = self.merge_adjacent_fixations(df)
        df = self.compute_eye_movement_duration(df)
        df = self.discard_short_fixation(df)
        df = self.compute_fixation_centroids(df)
        fixation_df = self.compute_fixation_df(df)
        return df,fixation_df
