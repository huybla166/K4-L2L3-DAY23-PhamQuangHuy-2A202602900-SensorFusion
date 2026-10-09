"""Extended Kalman filter helpers for 6D constant-velocity motion.

Part E supplies prediction and correction for docs/HUONG_DAN_KY_THUAT.md §2.
Read the shared time step and process-noise settings with get_tracking_params().
"""

from __future__ import annotations

from typing import Any
from typing import Optional

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray

# vi: Gợi ý module — params = get_tracking_params() sau khi import ở trên.


def build_F(dt: Optional[float] = None) -> Matrix:
    """Build the constant-velocity state transition matrix F.

    Args:
        dt: Time step in seconds; default from tracking params.

    Returns:
        6x6 state transition matrix as ``np.matrix``.
    """
    params = get_tracking_params()
    if dt is None:
        dt = params.dt
    F = np.identity(params.dim_state)
    # Position integrates velocity over one step: p_k+1 = p_k + v_k * dt.
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    return np.asmatrix(F)


def build_Q(dt: Optional[float] = None, q: Optional[float] = None) -> Matrix:
    """Build the process noise covariance matrix Q.

    Args:
        dt: Time step; default from tracking params.
        q: Process noise scale; default from tracking params.

    Returns:
        6x6 process noise matrix.
    """
    params = get_tracking_params()
    if dt is None:
        dt = params.dt
    if q is None:
        q = params.q
    # Lab model: independent diagonal process noise growing linearly with dt.
    return np.asmatrix(np.identity(params.dim_state) * (dt * q))


def ekf_predict(
    x: Matrix,
    P: Matrix,
    F: Optional[Matrix] = None,
    Q: Optional[Matrix] = None,
) -> tuple[Matrix, Matrix]:
    """Predict state and covariance one time step forward.

    Args:
        x: State vector (6x1).
        P: State covariance (6x6).
        F: Optional transition matrix; build via ``build_F`` if None.
        Q: Optional process noise; build via ``build_Q`` if None.

    Returns:
        Tuple ``(x_pred, P_pred)``.
    """
    F = np.asmatrix(build_F() if F is None else F)
    Q = np.asmatrix(build_Q() if Q is None else Q)
    x = np.asmatrix(x)
    P = np.asmatrix(P)
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def innovation(x: Matrix, meas: Any) -> Matrix:
    """Compute the measurement residual (innovation) gamma.

    Args:
        x: Predicted state.
        meas: Measurement with ``z`` and ``sensor.get_hx(x)``.

    Returns:
        Innovation vector ``z - h(x)``.
    """
    return np.asmatrix(meas.z) - np.asmatrix(meas.sensor.get_hx(x))


def innovation_covariance(P: Matrix, meas: Any, H: Matrix) -> Matrix:
    """Compute the innovation covariance S = H P H' + R.

    Args:
        P: State covariance.
        meas: Measurement with ``R``.
        H: Measurement Jacobian.

    Returns:
        Innovation covariance matrix S.
    """
    H = np.asmatrix(H)
    return H @ np.asmatrix(P) @ H.T + np.asmatrix(meas.R)


def ekf_update(x: Matrix, P: Matrix, meas: Any) -> tuple[Matrix, Matrix]:
    """Apply an EKF measurement update and return updated state and covariance.

    Args:
        x: Prior state.
        P: Prior covariance.
        meas: Associated measurement.

    Returns:
        Tuple ``(x_upd, P_upd)``.
    """
    x = np.asmatrix(x)
    P = np.asmatrix(P)
    # Linearise h at the prior: constant for lidar, a Jacobian for the camera.
    H = np.asmatrix(meas.sensor.get_H(x))
    gamma = innovation(x, meas)
    S = innovation_covariance(P, meas, H)
    K = P @ H.T @ np.linalg.inv(S)
    x_upd = x + K @ gamma
    P_upd = (np.identity(P.shape[0]) - K @ H) @ P
    return x_upd, np.asmatrix(P_upd)
