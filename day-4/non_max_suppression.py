import numpy as np

def non_max_suppression_interpolated(mag, ang):
    """
    Interpolated non-maximum suppression.
    mag  : 2D array of gradient magnitudes
    ang  : 2D array of gradient angles in radians (np.arctan2 output)
    Returns: 2D array same shape as mag (suppressed)
    """
    H, W = mag.shape[:2]
    out = np.zeros_like(mag, dtype=mag.dtype)

    # Convert angle to degrees and make range [0,180)
    ang_deg = (np.rad2deg(ang) + 180) % 180

    # offsets for neighbors; we'll compute projected neighbors
    for i in range(1, H-1):
        for j in range(1, W-1):
            theta = ang_deg[i, j]

            # direction components (unit vector)
            # convert degrees to radians for cos/sin
            t = np.deg2rad(theta)
            dx = np.cos(t)
            dy = np.sin(t)

            # sample positions in +direction and -direction (float coords)
            x_pos = j + dx
            y_pos = i + dy
            x_neg = j - dx
            y_neg = i - dy

            # bilinear interpolation helper
            def bilinear_sample(img, y, x):
                x0 = int(np.floor(x)); x1 = x0 + 1
                y0 = int(np.floor(y)); y1 = y0 + 1

                # boundary check
                if x0 < 0 or x1 >= W or y0 < 0 or y1 >= H:
                    return 0.0

                Ia = img[y0, x0]
                Ib = img[y0, x1]
                Ic = img[y1, x0]
                Id = img[y1, x1]

                wa = (x1 - x) * (y1 - y)
                wb = (x - x0) * (y1 - y)
                wc = (x1 - x) * (y - y0)
                wd = (x - x0) * (y - y0)

                return Ia*wa + Ib*wb + Ic*wc + Id*wd

            m = mag[i, j]
            m_pos = bilinear_sample(mag, y_pos, x_pos)
            m_neg = bilinear_sample(mag, y_neg, x_neg)

            if (m >= m_pos) and (m >= m_neg):
                out[i, j] = m
            else:
                out[i, j] = 0.0

    return out