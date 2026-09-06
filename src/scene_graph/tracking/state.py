import numpy as np

class KalmanState:
    """6D Constant-Velocity Kalman Filter State."""
    
    def __init__(self, position: np.ndarray, initial_cov_pos: float = 0.1, initial_cov_vel: float = 1.0):
        # State vector: [x, y, z, vx, vy, vz]^T
        self.x = np.zeros(6)
        self.x[:3] = position
        
        # Covariance matrix P (6x6)
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = np.eye(3) * initial_cov_pos
        self.P[3:, 3:] = np.eye(3) * initial_cov_vel
        
    def predict(self, dt: float, q_std: float):
        """Predict step of Kalman Filter.
        
        Args:
            dt: Time step in seconds.
            q_std: Process noise standard deviation (acceleration).
        """
        if dt <= 0:
            return
            
        # State transition matrix F
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
        # Process noise covariance Q
        dt2 = dt**2
        dt3 = dt**3 / 2.0
        dt4 = dt**4 / 4.0
        
        q_var = q_std**2
        Q = np.zeros((6, 6))
        
        Q[0:3, 0:3] = np.eye(3) * dt4 * q_var
        Q[0:3, 3:6] = np.eye(3) * dt3 * q_var
        Q[3:6, 0:3] = np.eye(3) * dt3 * q_var
        Q[3:6, 3:6] = np.eye(3) * dt2 * q_var
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        
    def update(self, measurement: np.ndarray, R: np.ndarray):
        """Update step of Kalman Filter.
        
        Args:
            measurement: 3D position measurement (3,)
            R: Measurement noise covariance (3x3)
        """
        # Observation matrix H
        H = np.zeros((3, 6))
        H[:, 0:3] = np.eye(3)
        
        # Innovation y = z - Hx
        y = measurement - H @ self.x
        
        # Innovation covariance S = HPH^T + R
        S = H @ self.P @ H.T + R
        
        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv
        except np.linalg.LinAlgError:
            return
            
        # Update state
        self.x = self.x + K @ y
        
        # Joseph form update for numerical stability
        I = np.eye(6)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
    @property
    def position(self) -> np.ndarray:
        return self.x[:3]
        
    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6]
        
    @property
    def position_covariance(self) -> np.ndarray:
        return self.P[:3, :3]
