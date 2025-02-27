import numpy as np
import preprocessing
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter, find_peaks


class Settings():
    def __init__(self):
        self.Fs = 60  # Sample rate of eye tracker
        self.gap_dur = 400  # Max gaps between period of data loss, interpolate smaller gaps
        self.min_amplitude = 0.1  # % of fully open eye (0.1 - 10%)
        self.min_separation = 100  # Min separation between blinks
        self.debug = False

        self.filter_length = 25  # in ms
        self.width_of_blink = 15  # in ms width of peak to initially detect
        self.min_blink_dur = 30  # Reject blinks shorter than 30 ms


class BlinkDetector:

    def __init__(self, settings):
        self.settings = settings

    def _binary2bounds(self, binary):
        """Find blink onsets and offsets."""
        d = np.diff(np.hstack((0, binary, 0)))
        onsets = np.where(d == 1)[0]
        offsets = np.where(d == -1)[0] - 1

        # Ensure onsets and offsets match
        if len(offsets) > len(onsets):
            offsets = offsets[1:] if onsets[0] > offsets[0] else offsets[:-1]
        elif len(offsets) < len(onsets):
            onsets = onsets[1:] if onsets[0] > offsets[0] else onsets[:-1]

        return onsets + 1, offsets

    def _merge_blinks(self, blink_onsets, blink_offsets, min_dur, min_separation,
                     additional_params=[]):
        """

        Merges blinks close together, and removes short blinks
        Args:
            blink_onsets (list): onsets of blinks (ms)
            blink_offsets (list): offsets of blinks (ms)
            min_dur (int): minimum duration of blink (ms)
            min_separation (int): minimal duration between blinks (ms, those with smaller are merged)

        Returns:
            blinks (list): list with onset, offset, duration

        """


        # Merge blink candidate close together, and remove short, isolated ones
        new_onsets = []
        new_offsets = []
        new_parameters = []
        change_onset = True

        for i, onset in enumerate(blink_onsets):
            # print(i, blink_onsets[i])
            if change_onset:
                temp_onset = blink_onsets[i]

            if i < len(blink_onsets) - 1:
                if ((blink_onsets[i+1] - blink_offsets[i])) < min_separation:

                    # if change_onset:
                    #     temp_onset = onsets[i]
                    change_onset = False
                else:
                    change_onset = True

                    # Remove blink with too short duration
                    if ((blink_offsets[i] - temp_onset)) < min_dur:
                        continue

                    new_onsets.append(temp_onset)
                    new_offsets.append(blink_offsets[i])
                    if len(additional_params) > 0:
                        new_parameters.append(additional_params[i, :])
            else:

                # # Remove blink with too short duration
                if ((blink_offsets[i] - temp_onset)) < min_dur:
                    continue

                new_onsets.append(temp_onset)
                new_offsets.append(blink_offsets[i])
                if len(additional_params) > 0:
                    new_parameters.append(additional_params[i, :])

        # Compute durations and convert to array
        blinks = []
        for i in range(len(new_onsets)):
            dur = new_offsets[i] - new_onsets[i]

            if len(additional_params) > 0:
                blinks.append([new_onsets[i], new_offsets[i], dur] +
                              list(new_parameters[i]))
            else:
                blinks.append([new_onsets[i], new_offsets[i], dur])

        # print(len(blink_onsets), len(new_onsets))

        return blinks


    def blink_detector_eo(self, t, eye_openness_signal, Fs, gap_dur=30,
                       filter_length=25,
                       width_of_blink=15,
                       min_separation=100,
                       plot_on=True):
        """

        Args:
            t - time in ms
            eye_openness_signal (1d numpy array): eye openness signal for left or right eye
            Fs (int): sampling frequency of the eo data
            gap_dur (int): interpolate gaps shorter than this duration (ms)
            filter_length (int): length of SG filter (ms)
            width_of_blink (int): min width of blink (ms)
            min_separation (int): min separation between blink peaks (ms)
            plot_on (boolean): plot the results?

        Returns:
            df - pandas dataframe with blink parameters

        """

        ms_to_sample = Fs / 1000
        sample_to_ms = 1000 / Fs


        # Assumes the eye is mostly open during the trial
        fully_open = np.nanmedian(eye_openness_signal, axis=0)
        min_amplitude = fully_open * self.settings.min_amplitude  # Equivalent to height in 'find_peaks'

        # detection parameters in samples
        distance_between_blinks = 1
        width_of_blink = width_of_blink * ms_to_sample
        filter_length = preprocessing.nearest_odd_integer(filter_length * ms_to_sample)
        filter_length = 3

        # Interpolate gaps
        eye_openness_signal = preprocessing.interpolate_nans(t, eye_openness_signal,
                                                              gap_dur=int(gap_dur))

        # Filter eyelid signal and compute
        eye_openness_signal_filtered = savgol_filter(eye_openness_signal, filter_length,2,
                                       mode='nearest')
        eye_openness_signal_vel = savgol_filter(eye_openness_signal, filter_length, 2,
                                           deriv=1,  mode='nearest') * Fs

        # Velocity threshold for on-, and offsets
        T_vel = stats.median_abs_deviation(eye_openness_signal_vel, nan_policy='omit') * 3

        # Turn blink signal into something that looks more like a saccade signal
        eye_openness_signal_inverse = (eye_openness_signal_filtered -
                                       np.nanmax(eye_openness_signal_filtered)) * -1
        peaks, properties = find_peaks(eye_openness_signal_inverse, height=None,
                                       distance=distance_between_blinks,
                                       width=width_of_blink)

        # Filter out not so 'prominent peaks'
        '''
        The prominence of a peak may be defined as the least drop in height
         necessary in order to get from the summit [peak] to any higher terrain.
        '''
        idx = properties['prominences'] > min_amplitude
        peaks = peaks[idx]
        for key in properties.keys():
            properties[key] = properties[key][idx]

        # Find peak opening/closing velocity by searching for max values
        # within a window from the peak
        blink_properties = []
        for i, peak_idx in enumerate(peaks):

            # Width of peak
            width = properties['widths'][i]

            ### Compute opening/closing velocity
            # First eye opening velocity (when eyelid opens after a blink)
            peak_right_idx = np.nanargmax(eye_openness_signal_vel[peak_idx:int(peak_idx + width)])
            peak_right_idx = np.nanmin([peak_right_idx, len(eye_openness_signal_vel)])
            idx_max_opening_vel = int(peak_idx + peak_right_idx)
            time_max_opening_vel = t[idx_max_opening_vel]
            opening_velocity = np.nanmax(eye_openness_signal_vel[peak_idx:int(peak_idx + width)])

            # Then eye closing velocity (when eyelid closes in the beginning of a blink)
            peak_left_idx = width - np.nanargmin(eye_openness_signal_vel[np.max([0, int(peak_idx - width)]):peak_idx]) + 1
            peak_left_idx = np.nanmax([peak_left_idx, 0])
            idx_max_closing_vel = int(peak_idx - peak_left_idx + 1)
            time_max_closing_vel = t[idx_max_closing_vel]
            closing_velocity = np.nanmin(eye_openness_signal_vel[np.max([0, int(peak_idx - width)]):peak_idx])

            # Identify on and offsets (go from peak velocity backward/forward)
            temp = eye_openness_signal_vel[idx_max_opening_vel:]
            if np.any(temp <= (T_vel / 3)):
                offset = np.where(temp <= (T_vel / 3))[0][0]
            else:
                offset = len(temp)

            # make sure the blink period stop when encountering nan-data
            # If it does, make the opening phase parameters invalid
            if np.any(np.isnan(temp)):
                offset_nan = np.where(np.isnan(temp))[0][0]
                offset = np.min([offset, offset_nan])

            offset_idx = int(idx_max_opening_vel + offset - 1)

            temp = np.flip(eye_openness_signal_vel[:idx_max_closing_vel])
            if np.any(temp >= -T_vel):
                onset = np.where(temp >= -T_vel)[0][0]
            else:
                onset = 0

            if np.any(np.isnan(temp)):
                onset_nan = np.where(np.isnan(temp))[0][0]
                onset = np.min([onset, onset_nan])

            onset_idx = int(idx_max_closing_vel - onset)


            # Compute openness at onset, peak, and offset
            openness_at_onset = eye_openness_signal_filtered[onset_idx]
            openness_at_offset = eye_openness_signal_filtered[offset_idx]
            openness_at_peak = eye_openness_signal_filtered[peak_idx]

            # Compute amplitudes for closing and opening phases
            closing_amplitude = np.abs(openness_at_onset - openness_at_peak)
            opening_amplitude = np.abs(openness_at_offset - openness_at_peak)

            distance_onset_peak_vel = np.abs(eye_openness_signal_filtered[onset_idx] -
                                             eye_openness_signal_filtered[idx_max_closing_vel]) # mm
            timediff_onset_peak_vel = np.abs(onset_idx - idx_max_closing_vel) * sample_to_ms # ms

            # Onset and peak cannot be too close in space and time
            if (distance_onset_peak_vel < 0.1) or (timediff_onset_peak_vel < 10):
                if self.settings.debug:
                    print('Peak to close to onset')
                continue

            if np.min([opening_velocity, np.abs(closing_velocity)]) < (T_vel * 2):
                if self.settings.debug:
                    print('Blink velocity too low')
                continue

            blink_properties.append([t[onset_idx],
                                     t[offset_idx],
                                     t[offset_idx] - t[onset_idx],
                                     t[peak_idx],
                                     openness_at_onset, openness_at_offset,
                                     openness_at_peak,
                                     time_max_opening_vel,
                                     time_max_closing_vel,
                                     opening_velocity, closing_velocity,
                                     opening_amplitude, closing_amplitude])

        # Are there any blinks found?
        if len(blink_properties) == 0:
            bp = []
        else:

            # Merge blinks too close together in time
            blink_temp = np.array(blink_properties)
            blink_onsets = blink_temp[:, 0]
            blink_offsets = blink_temp[:, 1]

            bp =  self._merge_blinks(blink_onsets, blink_offsets, width_of_blink, min_separation,
                             additional_params=blink_temp[:, 3:])

        # Convert to dataframe
        df = pd.DataFrame(bp,
                          columns=['onset', 'offset', 'duration',
                                   'time_peak',
                                   'openness_at_onset',
                                   'openness_at_offset',
                                   'openness_at_peak',
                                   'time_peak_opening_velocity',
                                   'time_peak_closing_velocity',
                                   'peak_opening_velocity',
                                   'peak_closing_velocity',
                                   'opening_amplitude',
                                   'closing_amplitude'])

        df.iloc[df.openness_at_peak < 0] = 0

        return df, eye_openness_signal_vel

if __name__ == "__main__":
    file_path = "csv results/test2_raw_svg.xlsx" 
    df = pd.read_excel(file_path)
    selected_columns = ["Recording timestamp","Eye openness left","Eye openness right","Eye movement type"]
    #df= df[~df['Event'].isin(['ImageStimulusStart', 'ImageStimulusEnd',"RecordingStart","RecordingEnd"])].reset_index(drop=True)
    df = df[selected_columns].copy()

    # compute average direction and openness of both eyes
    df["Eye openness"] = df.apply(lambda row: preprocessing.compute_gaze_direction(row["Eye openness left"], row["Eye openness right"]), axis=1)
    df.loc[df["Eye movement type"] == "EyesNotFound"] = np.nan
    #df = df.iloc[5:].reset_index(drop=True)
    eo_signal = df["Eye openness"].values
    time = df["Recording timestamp"].values
    settings = Settings()
    bd = BlinkDetector(settings)
    df_out, eye_openness_signal_vel = bd.blink_detector_eo(time, eo_signal, settings.Fs, filter_length=settings.filter_length,
                                                             gap_dur=settings.gap_dur,
                                                             width_of_blink=settings.width_of_blink,
                                                             min_separation=settings.min_separation)

    print(df_out)
