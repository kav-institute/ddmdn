import numpy as np


class Ego_Transformer:
    """
    Ego-vehicle coordinate transformer for 2D points.
    Args:
        translation_vector (np.ndarray [2]): Translation vector to shift coordinates.
        rotation_angle (float): Rotation angle in radians.
    """
    
    def __init__(self, translation_vector, rotation_angle):
        """
        Initialize the ego transformation parameters.
        Args:
            translation_vector (np.ndarray [2]): Translation vector for the ego frame.
            rotation_angle (float): Rotation angle in radians.
        """
        
        self.translation_vector = translation_vector
        self.rotation_angle = rotation_angle
        
        return
    
    
    def reset(self, translation_vector, rotation_angle):
        """
        Reset translation and rotation parameters.
        Args:
            translation_vector (np.ndarray [2]): New translation vector.
            rotation_angle (float): New rotation angle in radians.
        """
        
        self.translation_vector = translation_vector
        self.rotation_angle = rotation_angle
        return
    
    
    def apply(self, X):
        """
        Apply ego transformation to world coordinates.
        Args:
            X (np.ndarray [n, 2]): Points in world frame.
        Returns:
            np.ndarray [n, 2]: Points transformed to ego frame.
        """
        
        R = self.build_R()
        X_t = X - self.translation_vector
        X_Rt = (R @ X_t.T).T
        
        return X_Rt
    
    
    def norm(self, X, size):
        """
        Normalize coordinates by a scaling factor.
        Args:
            X (np.ndarray or torch.Tensor [n, d]): Coordinates to normalize.
            size (float or np.ndarray): Scaling factor.
        Returns:
            Same type as X: Normalized coordinates.
        """
        
        X_normed = X / (size)
        return X_normed
    
    
    def denorm(self, X_normed, size):
        """
        Denormalize coordinates by a scaling factor.
        Args:
            X_normed (np.ndarray or torch.Tensor [n, d]): Normalized coordinates.
            size (float or np.ndarray): Scaling factor.
        Returns:
            Same type as X_normed: Denormalized coordinates.
        """
        
        X = X_normed * size
        return X
    
    
    def revert(self, X):
        """
        Revert ego transformation to original world coordinates.
        Args:
            X (np.ndarray [n, 2]): Points in ego frame.
        Returns:
            np.ndarray [n, 2]: Original coordinates before transformation.
        """
        
        R = self.build_R()
        R_inv = R.T
        X_Rinv = (R_inv @ X.T).T
        X_original = X_Rinv + self.translation_vector
        return X_original
    
    
    def build_R(self):
        """
        Build rotation matrix based on current angle.
        Returns:
            np.ndarray [2, 2]: Rotation matrix for self.rotation_angle.
        """
        
        # get identity matrices
        R = np.zeros((2, 2))
        R[0, 0] = np.cos(self.rotation_angle)
        R[0, 1] = -np.sin(self.rotation_angle)
        R[1, 0] = np.sin(self.rotation_angle)
        R[1, 1] = np.cos(self.rotation_angle)
        
        return R