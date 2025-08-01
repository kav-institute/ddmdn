import numpy as np


TARGET_CLASS_DICT = {'pedestrian': 0, 'bicycle': 1, 'scooter': 2, 'skater': 3, 'cart': 4, 'motorbike': 5, 'car': 6, 'truck_bus': 7, 'ball': 8, 'stroller': 9, 'wheelchair': 10}


TARGET_SEG_DICT = {
                    'unlabeled': 0,     # out of map or unlabeled/unknown
                    'blocking': 1,      # buildings, walls, fences, ...
                    'road': 2,          # road and road marks
                    'sidewalk': 3,      # vru prefered
                    'bikelane': 4,      # cyclist prefered
                    'crosswalk': 5,     # vru crosswalk
                    'vegetation': 6,    # grass, bushes, trees
                    'terrain': 7        # walkable, but not prefered
                    }


IMPTC_SEG_LSA_POLYS = {
    "f1": [[574, 782], [783, 629], [804, 651], [828, 668], [855, 679], [878, 684], [903, 686], [610, 901], [609, 884], [607, 863], [600, 835], [592, 814], [579, 790]],
    "f2": [[983, 675], [1005, 663], [1087, 598], [1389, 989], [1306, 1056], [1292, 1074]],
    "f3": [[985, 1412], [1251, 1202], [1253, 1222], [1258, 1244], [1270, 1268], [1283, 1287], [1301, 1309], [1076, 1486], [1042, 1447], [1026, 1431], [1006, 1418]]
}


IMPTC_MAP_LSA_POLYS = { 
    "f1": [[374, 582], [583, 429], [604, 451], [628, 468], [655, 479], [678, 484], [703, 486], [410, 701], [409, 684], [407, 663], [400, 635], [392, 614], [379, 590]],
    "f2": [[783, 475], [805, 463], [887, 398], [1189, 789], [1106, 856], [1092, 874]],
    "f3": [[785, 1212], [1051, 1002], [1053, 1022], [1058, 1044], [1070, 1068], [1083, 1087], [1101, 1109], [876, 1286], [842, 1247], [826, 1231], [806, 1218]]
        }


def get_key(d, v):
    
    k = [k for k, val in d.items() if val == v][0]
    return k


def check_timestamps(timestamps, frame_step=1):
    
    sublists = []
    current_sublist = [timestamps[0]]
    
    for i in range(1, len(timestamps)):
        
        if timestamps[i] - timestamps[i-1] == frame_step:
            
            current_sublist.append(timestamps[i])
            
        else:
            
            sublists.append(current_sublist)
            current_sublist = [timestamps[i]]
            
    sublists.append(current_sublist)
    
    return sublists


def check_velocity_exceedance(tracks, dt, max_velo, exceeding_limit):
    
    filtered_tracks = []
    filter_cnt = 0
    
    for T in tracks:
        
        positions = T[...,3:5].astype(np.float32)
        
        # Compute velocity magnitudes using vectorized operations
        differences = np.diff(positions, axis=0)  # Shape (n-1, 2)
        distances = np.linalg.norm(differences, axis=1)  # Euclidean distance for each interval
        velocities = distances / dt  # Velocity for each interval
        
        # Determine the proportion of intervals exceeding max_velocity
        exceedances = velocities > max_velo
        proportion_exceeding = np.mean(exceedances)
        
        # Check
        if proportion_exceeding <= exceeding_limit:
            filtered_tracks.append(T)
        else:
            filter_cnt += 1
    
    return filtered_tracks, filter_cnt


def incomplete_usability_check(track, obs_len):
    
    # check if at least input- and next two ouput horizon steps have valid measurements, else this object is not affecting the current sequence at this timestamp
    input = track[:obs_len]
    future = track[obs_len:]
    return np.all(~np.isnan(input[-obs_len:, :])) and np.all(~np.isnan(future[:2, :]))
