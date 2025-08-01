import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def resample_tracks(trajectories, obs_len, seq_len, original_rate, resample_rate, with_smoothing=False, smoothing_window=5, smoothing_polyorder=2):
    """
    Resamples a batch of 2D trajectories to a different sampling rate, handling NaNs.
    
    Parameters:
        trajectories (ndarray): Array of shape (batch_size, N, 2) where each trajectory has N points.
        original_rate (float): Original sampling rate in Hz (e.g., 25).
        target_rate (float): Target sampling rate in Hz.
        
    Returns:
        resampled_trajectories (ndarray): Resampled trajectories of shape (batch_size, M, 2).
    """
    
    # Calculate original and target time vectors
    original_duration = seq_len * original_rate
    original_times = np.linspace(0, original_duration, seq_len)
    num_target_samples = int(original_duration / resample_rate)
    target_times = np.linspace(0, original_duration, num_target_samples)
    
    for i in range(len(trajectories)):
        
        data = trajectories[i]
        valid_mask = ~np.isnan(data)
        
        # Check if there are valid points
        if np.any(valid_mask):
            valid_times = original_times[valid_mask[...,0]]
            valid_data = data[valid_mask[...,0]]
            
            # Interpolate the valid data
            x_interp = interp1d(valid_times, valid_data[:,0], kind='cubic', bounds_error=False, fill_value=np.nan)
            y_interp = interp1d(valid_times, valid_data[:,1], kind='cubic', bounds_error=False, fill_value=np.nan)
            
            resampled_x = x_interp(target_times)
            resampled_y = y_interp(target_times)
            
            if with_smoothing:
                
                resampled_x = savgol_filter(resampled_x, smoothing_window, smoothing_polyorder)
                resampled_y = savgol_filter(resampled_y, smoothing_window, smoothing_polyorder)
                
            # Store the resampled data
            resampled_track = np.stack([resampled_x, resampled_y], axis=-1)
                
            
    observation_length = int((obs_len * original_rate) / resample_rate)
    return resampled_track, observation_length