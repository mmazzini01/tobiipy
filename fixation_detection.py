import pandas as pd
import numpy as np
import blink_detection

class FixationDetector():
    def __init__(self, settings,fix,nan_values,blink_settings):
        self.settings = settings
        self.fix = fix
        self.nan_values = nan_values
        self.blink_detection = blink_detection.BlinkDetector(blink_settings)

    def noise_reduction(self,df):
        # If noise_reduction is None or False, return df unchanged
        if not self.settings.noise_reduction:
            return df
        df_filtered = df.copy()
        if self.settings.noise_reduction_window <= 0 or self.settings.noise_reduction_window % 2 == 0:
            raise ValueError("Noise reduction window must be a positive odd number.")
        window_size = self.settings.noise_reduction_window // 2
        for i in range(1, len(self.fix) - 1):  # Avoid first and last indices
            id = self.fix[i]
            prev = self.fix[i - window_size]
            post = self.fix[i + window_size]
            window = df.loc[prev:post]
            if not window[['Gaze point X', 'Gaze point Y', 'Gaze point Z']].isna().values.any():
                for col in df.columns:
                    if col not in ["Eye movement type", "Recording timestamp","Eye openness left","Eye openness right"]:
                        if self.settings.noise_reduction == 'median':
                            value = np.median([df.at[prev, col], df.at[id, col], df.at[post, col]])
                        elif self.settings.noise_reduction == 'average':
                            value = np.mean([df.at[prev, col], df.at[id, col], df.at[post, col]])
                        df_filtered.at[id, col] = value
        return df_filtered
    
    def compute_angle(self,array1,array2, matrix = None):
        if not matrix:   
            dot = np.dot(array1, array2)
            norm_start = np.linalg.norm(array1)
            norm_end   = np.linalg.norm(array2)
            theta = np.arccos(np.clip(dot / (norm_start * norm_end),-1,1))
            theta_deg = np.degrees(theta)
            return theta_deg
        else:
            dot = np.sum(array1 * array2, axis=1)  
            norm_start = np.linalg.norm(array1, axis=1)  
            norm_end   = np.linalg.norm(array2, axis=1)  
            theta = np.arccos(np.clip(dot / (norm_start * norm_end), -1, 1)) 
            theta_deg = np.degrees(theta)  
            return theta_deg 
    
    def compute_velocity(self,df):
        df['Velocity'] = np.nan
        if not self.settings.velocity_window:
            for i in range(1,len(self.fix)):
                index_fix = self.fix[i]
                index_prev = index_fix -1
                if index_prev not in self.nan_values:
                    sample_prev = df.iloc[index_prev]
                    sample_middle= df.iloc[index_fix]
                    dt = (sample_middle['Recording timestamp'] - sample_prev['Recording timestamp']) / 1000 # Convert ms to sec
                    direction_start = np.array(sample_prev[['Gaze point X', "Gaze point Y",'Gaze point Z']].astype(float).round(2))
                    eye_middle = np.array(sample_middle[["Eye position X", "Eye position Y", "Eye position Z"]].astype(float).round(2))
                    direction_end = np.array(sample_middle[['Gaze direction X', 'Gaze direction Y', 'Gaze direction Z']].astype(float).round(2))
                    v1 = direction_start-eye_middle

                    theta_deg = self.compute_angle(v1,direction_end,matrix=None)
                    ang_vel = theta_deg / 0.017
                    df.at[index_fix, 'Velocity'] = ang_vel
        else:
            for i in range(1,len(self.fix)):
                index_fix = self.fix[i]
                index_prev = index_fix -1
                if index_prev not in self.nan_values:
                    sample_prev = df.iloc[index_prev]
                    sample_middle= df.iloc[index_fix]
                    dt = (sample_middle['Recording timestamp'] - sample_prev['Recording timestamp']) / 1000
                    '''
                    direction_start =np.array(sample_prev[["Gaze direction X", "Gaze direction Y", "Gaze direction Z"]].astype(float))
                    eye_middle = np.array(sample_middle[["Gaze direction X", "Gaze direction Y", "Gaze direction Z"]].astype(float))
                    direction_end = np.array(sample_post[["Gaze direction X", "Gaze direction Y", "Gaze direction Z"]].astype(float))
                    '''
                    direction_start = np.array(sample_prev[['Gaze point X', "Gaze point Y",'Gaze point Z']].astype(float).round(2))
                    eye_middle = np.array(sample_middle[["Eye position X", "Eye position Y", "Eye position Z"]].astype(float).round(2))
                    direction_end = np.array(sample_middle[['Gaze point X', "Gaze point Y",'Gaze point Z']].astype(float).round(2))
                    v1 = direction_start-eye_middle
                    v2 = direction_end -eye_middle
                    v1 = np.round(v1,2)
                    v2 = np.round(v2,2)
                    ang1 = self.compute_angle(v1,v2)
                    ang_vel = ang1 / 0.017
                    df.at[index_fix, 'Velocity'] = ang_vel
        return df
    
    def classify_movement(self,df):
        df['Classified Movement'] = df['Eye movement type']  # Preserve existing labels
        filtered_rows = ~pd.isna(df['Velocity']) & (df['Velocity'] >= self.settings.velocity_threshold)
        df.loc[filtered_rows, 'Classified Movement'] = 'Saccade'
        # if sample is after a nan, take the label of the next sample
        df['Classified Movement'] = df['Classified Movement'].where(~(pd.isna(df['Velocity']) & (df['Classified Movement'] == 'Fixation')
                                                                      ), df['Classified Movement'].shift(-1))
        df.at[0,"Classified Movement"] = "Fixation"
        return df


    def assign_unique_group_number(self,df):
         #Assign group numbers
        df['Group'] = (df['Classified Movement'] != df['Classified Movement'].shift()).cumsum()
        return df
    
    def assign_movement_index(self, row):
        """Assigns an incremental group number to each label when a new group appears."""
        label = row['Classified Movement']  # Use the correct column name
        group = row['Group']

        if group > self.settings.label_counter.get('id'):
            # Increment label-specific counter
            self.settings.label_counter[label] = self.settings.label_counter.get(label, 0) + 1
            self.settings.label_counter['id'] +=1

        return self.settings.label_counter[label]
    
    def compute_eye_movement_duration(self,df):

        first_timestamps = df.groupby('Group')['Recording timestamp'].first()
        last_timestamps = df.groupby('Group')['Recording timestamp'].last()

        # Step 3: Compute the start time (midpoint of previous last and current first)
        prev_last_timestamps = last_timestamps.shift(1)  # Last timestamp of previous group
        start_times = (prev_last_timestamps + first_timestamps) // 2

        # Step 4: Compute the end time (midpoint of current last and next first)
        next_first_timestamps = first_timestamps.shift(-1)  # First timestamp of next group
        end_times = (last_timestamps + next_first_timestamps) // 2

        start_times.iloc[0] = first_timestamps.iloc[0]  # First group starts at its first timestamp
        end_times.iloc[-1] = last_timestamps.iloc[-1]  # Last group ends at its last timestamp

        # First group duration = (First timestamp of next group - First timestamp of first group)
        start_times.iloc[0] = first_timestamps.iloc[0]
        end_times.iloc[0] = first_timestamps.iloc[1]  # Next group's first timestamp

        # Last group duration = (Last timestamp of last group - Last timestamp of previous group)
        start_times.iloc[-1] = last_timestamps.iloc[-2]  # Previous group's last timestamp
        end_times.iloc[-1] = last_timestamps.iloc[-1]

        # Step 6: Compute the final duration
        durations = end_times - start_times

        # Step 7: Map the computed durations back to the original DataFrame
        df['Eye movement duration'] = df['Group'].map(durations)
        return df

        
    def merge_adjacent_fixations(self,df):
        df = self.assign_unique_group_number(df)
        if self.settings.merge_fixations:

            """
            Merges adjacent fixations based on time gap and angular velocity.
            """
            df = df.copy()
            df_fix = df[df['Classified Movement'] == "Fixation"].copy()

            # Step 3: Compute time differences between consecutive fixation groups
            group_last_timestamps = df_fix.groupby("Group")['Recording timestamp'].last()
            group_first_timestamps = df_fix.groupby("Group")['Recording timestamp'].first()
            time_gaps = group_first_timestamps.shift(-1) - group_last_timestamps  # Time gap to next group

            # Step 4: Compute average eye position between consecutive fixations
            eye_last = df_fix.groupby("Group")[["Eye position X","Eye position Y","Eye position Z"]].last().to_numpy()
            eye_first = df_fix.groupby("Group")[["Eye position X","Eye position Y","Eye position Z"]].first().shift(-1).to_numpy()
            
            print(eye_last.shape)
            avg_eye= np.mean([eye_first,eye_last], axis=0)

            # Step 5: Compute visual angles between consecutive fixations
            gaze_last = df_fix.groupby("Group")[["Gaze point X","Gaze point Y","Gaze point Z"]].last().to_numpy()
            gaze_first = df_fix.groupby("Group")[["Gaze point X","Gaze point Y","Gaze point Z"]].first().shift(-1).to_numpy()

            dir_to_last = gaze_last - avg_eye
            dir_to_first = gaze_first -avg_eye
            angles = self.compute_angle(dir_to_first,dir_to_last, matrix=True)

            # Step 6: Determine which fixations should be merged
            merge_groups = (time_gaps < self.settings.merge_max_time) & (angles < self.settings.merge_max_angle)

            # Step 7: Map merge decisions back to the original DataFrame
            merge_flags = df_fix['Group'].map(merge_groups.fillna(False))

            df["Merged"] = df["Group"].map(merge_flags.fillna(False))
            df["Merged"] = df["Merged"].fillna(False).astype(bool)

            df.loc[df["Merged"], "Classified Movement"] = "Fixation"
            # assign new group numbers
            df.drop(columns=['Group'], inplace=True)
            df = self.assign_unique_group_number(df)
            # Drop 'Merged' column (optional)
            #df.drop(columns=['Merged'], inplace=True)
        df['Label Group'] = df.apply(self.assign_movement_index, axis=1)
        return df
    
    def discard_short_fixation(self,df):
        if self.settings.discard_short_fixations:
            df.loc[(df["Classified Movement"] == "Fixation") & (df["Eye movement duration"] < self.settings.min_fixation_duration), "Classified Movement"] = "Unclassified"
            return df
        else:
            return df
        
    def blink_call(self,df):
        blink_class = self.blink_detection
        if blink_class.settings.blink_detection:
            t =((df["Recording timestamp"].values - df["Recording timestamp"][0])).round(0)
            eye_op_left = df["Eye openness left"].values
            eye_op_right = df["Eye openness right"].values
            blink_df = blink_class.blink_detector_eo(t,eye_op_left,eye_op_right)
            print(blink_df)
            gap = t-t[0]
            blink_starts = blink_df['onset'].values[:, None]  
            blink_ends = blink_df['offset'].values[:, None]   
            # For each timestamp, check if it's in any blink interval
            mask = ((gap >= blink_starts) & (gap < blink_ends)).any(axis=0)

            df.loc[mask, 'Classified Movement'] = 'Blink'
            df["Gap"] = gap
            
            return df
            
        else:
            return df

    
    def fixation_detector(self,df):
        df = self.noise_reduction(df)
        df = self.compute_velocity(df)
        df = self.classify_movement(df)
        df = self.blink_call(df)
        df = self.merge_adjacent_fixations(df)
        df = self.compute_eye_movement_duration(df)
        df = self.discard_short_fixation(df)
        return df
    

