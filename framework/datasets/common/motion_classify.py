import numpy as np

# create a mapping dictionary for movement labels
MOTION_DICT = {
    0: 'standing',
    1: 'starting',
    2: 'stopping',
    3: 'straight',
    4: 'light_left',
    5: 'light_right',
    6: 'strong_left',
    7: 'strong_right'
}

# create a mapping dictionary for movement labels
DIRECTION_DICT = {
    0: 'strong_right',
    1: 'strong_right',
    2: 'strong_right',
    3: 'strong_right',
    4: 'strong_right',
    5: 'strong_right',
    6: 'strong_right',
    7: 'light_right',
    8: 'light_right',
    9: 'straight',
    10: 'straight',
    11: 'light_left',
    12: 'light_left',
    13: 'strong_left',
    14: 'strong_left',
    15: 'strong_left',
    16: 'strong_left',
    17: 'strong_left',
    18: 'strong_left',
    19: 'strong_left',
    20: 'strong_left',
    21: 'strong_left',
    22: 'strong_left',
    23: 'strong_left',
    24: 'strong_left',
    25: 'strong_left',
    26: 'strong_left',
    27: 'light_left',
    28: 'light_left',
    29: 'straight',
    30: 'straight',
    31: 'light_right',
    32: 'light_right',
    33: 'strong_right',
    34: 'strong_right',
    35: 'strong_right',
    36: 'strong_right',
    37: 'strong_right',
    38: 'strong_right',
    39: 'strong_right'
}

class Motion_Classifier:
    
    def __init__(self, standing_threshold):
        
        self.coridor = 2
        self.standing_threshold = standing_threshold
        return
        
        
    def classify(self, T, obs_len):
        
        # Split into past and future
        X = T[:obs_len].copy()
        Y = T[obs_len:].copy()
        
        # Compute translation vector
        translation_vector = X[-1].copy()
        
        # Identify all past and future velocities above threshold
        past_walking_flags = list(False for _ in range(len(X)-1))
        future_walking_flags = list(False for _ in range(len(Y)-1))
        
        # Check past movement
        for i in range(len(X)-1):
            
            vel = X[i+1] - X[i]
            
            if np.linalg.norm(vel) >= self.standing_threshold:
                past_walking_flags[i] = True
                
        # Check future movement
        for i in range(len(Y)-1):
        
            vel = Y[i+1] - Y[i]
            
            if np.linalg.norm(vel) >= self.standing_threshold:
                future_walking_flags[i] = True
                
        # Classify motion state
        # No motion in the past => standing
        if all(not value for value in past_walking_flags):
            
            motion_state = MOTION_DICT[0]
            
            # Check if there is movement in the future
            idx = next((i for i, val in enumerate(future_walking_flags) if val), None)
            
            if idx is not None: 
                
                heading_vector = translation_vector - Y[idx]
                heading_factor = -1
                
            else:
            
                heading_vector = np.array([0.0, 1.0])
                heading_factor = 1
            
        # Full motion in the past => Walking
        elif all(past_walking_flags):
            
            motion_state = MOTION_DICT[3]
            heading_vector = translation_vector - X[-2]
            heading_factor = 1
        
        # In between - special cases
        else:
            
            # Signals
            upper_starting_continous_walking_length, _  = self.check_continuous_sequence(lst=past_walking_flags, target_value=True, from_end=True, slice_size=1.0)
            lower_starting_continous_walking_length, _ = self.check_continuous_sequence(lst=past_walking_flags, target_value=True, from_end=False, slice_size=1.0)
            upper_stopping_continous_standing_length, stopping_continous_standing_idx = self.check_continuous_sequence(lst=past_walking_flags, target_value=False, from_end=True, slice_size=1.0)
            lower_stopping_continous_standing_length, _ = self.check_continuous_sequence(lst=past_walking_flags, target_value=False, from_end=False, slice_size=1.0)
            still_walking, last_walking_idx = self.check_minimum_percentage(lst=past_walking_flags, target_value=True, percentage=0.4)
            
            # Continous walking starting from the nearest part of the past => starting
            if self.coridor <= upper_starting_continous_walking_length <= (len(past_walking_flags) - self.coridor) and lower_starting_continous_walking_length <= self.coridor:
                
                motion_state = MOTION_DICT[1]
                heading_vector = translation_vector - X[-2]
                heading_factor = 1
                
            # Continous standing starting from the nearest part of the past => stopping
            elif self.coridor <= upper_stopping_continous_standing_length <= (len(past_walking_flags) - self.coridor) and lower_stopping_continous_standing_length <= self.coridor:
                
                motion_state = MOTION_DICT[2]
                heading_vector = translation_vector - X[stopping_continous_standing_idx]
                heading_factor = 1
                
            # At least mostly walking
            elif still_walking:
                
                motion_state = MOTION_DICT[3]
                heading_vector = translation_vector - X[last_walking_idx]
                heading_factor = 1
            
            # No clear behaviour => standing
            else:
                
                motion_state = MOTION_DICT[0]
                
                # Check if there is movement in the future
                idx = next((i for i, val in enumerate(future_walking_flags) if val), None)
                
                if idx is not None: 
                    
                    heading_vector = translation_vector - Y[idx]
                    heading_factor = -1
                    
                else:
                
                    heading_vector = np.array([0.0, 1.0])
                    heading_factor = 1
                
        # Compute rotation angle to align heading_vector with the positive y-axis
        heading_angle = np.arctan2(heading_vector[1], heading_vector[0])
        rotation_angle = (heading_factor * (np.pi / 2)) - heading_angle
        
        # Update motion state if walking
        if motion_state == MOTION_DICT[3]:
            
            # Get identity matrices
            R = np.zeros((2, 2))
            R[0, 0] = np.cos(rotation_angle)
            R[0, 1] = -np.sin(rotation_angle)
            R[1, 0] = np.sin(rotation_angle)
            R[1, 1] = np.cos(rotation_angle)
            
            # Apply translation and rotation to future part of trajectory
            Y_t = Y - translation_vector
            Y_Rt = (R @ Y_t.T).T
            
            # Get polar coordinates radius
            angles, _ = self.get_polar_coordinates(T=Y_Rt)
            
            # Get the sine and cosine of each angle
            sin_values = np.sin(angles)
            cos_values = np.cos(angles)
            
            # Calculate the bin number for each angle
            bins = ((np.fmod(np.rad2deg(np.arctan2(sin_values, cos_values)) + 360, 360)) / 9)
            result_bin = np.floor(np.median(bins[-np.round(len(bins)*0.25).astype(int):])).astype(int)
            motion_state = DIRECTION_DICT[result_bin]
            
        return motion_state, translation_vector, rotation_angle
    
    
    def get_polar_coordinates(self, T):
        
        # Get x and y values of all positions
        x = T[:, 0]
        y = T[:, 1]
        
        # Calculate differences from given reference position for each n
        dx = x - 0
        dy = y - 0
        
        # Calculate the polar angle and distance using atan2
        angles = np.arctan2(dy, dx)
        radius = np.hypot(dx, dy)
        
        return angles, radius
    
    
    def check_continuous_sequence(self, lst, target_value=1, from_end=True, slice_size=0.5):
        """
        Checks for a continuous sequence of the target_value in a percentage-based slice of the list.
        
        Parameters:
        - lst: List of boolean (0/1) values.
        - target_value: The value (0 or 1) to look for.
        - from_end: If True, slice starts from the end; if False, slice starts from the beginning.
        - slice_size: Float (0.0 to 1.0) defining the percentage size of the slice.
        
        Returns:
        - True if a valid sequence is found, False otherwise.
        """
        
        n = len(lst)
        count = 0
        
        # Calculate the size of the slice in terms of indices
        slice_length = int(slice_size * n)
        
        # Determine the start and end indices based on `from_end`
        if from_end:
            
            start_idx = max(0, n - slice_length)
            end_idx = n
            
        else:
            
            start_idx = 0
            end_idx = min(n, slice_length)
            
        # Slice the list
        search_area = lst[start_idx:end_idx]
        
        # Reverse the search area if `from_end` is True
        if from_end:
            
            search_area = reversed(search_area)
            
        # Count the continuous sequence of the target value
        for val in search_area:
            
            if val == target_value:
                
                count += 1
            else:
                
                break
            
        if from_end:
            
            idx = n - count -1
            
        else:
            
            idx = count
            
        # Check if the count satisfies the minimum size requirement
        return count, idx
    
    
    def check_minimum_percentage(self, lst, target_value=True, percentage=0.5):
        """
        Checks if at least a specified percentage of the list contains the target value.
        
        Parameters:
        - lst: List of boolean (0/1) values.
        - target_value: The value (True or False) to check for.
        - percentage: Float (0.0 to 1.0) defining the minimum percentage threshold.
        
        Returns:
        - True if at least the defined percentage of values are the target value, False otherwise.
        """
        
        n = len(lst)
        current_count = 0
        
        # Calculate the minimum number of required target values
        required_count = int(percentage * n)
        
        # Iterate through the list and count occurrences of the target value
        for idx, val in enumerate(lst):
            
            if val == target_value:
                
                current_count += 1
                
        # Check if the condition is fulfilled
        if current_count >= required_count:
            
            return True, idx-1
        
        # Condition not met
        else:
            
            return False, 0