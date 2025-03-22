import numpy as np
import math

def latlon(XYZ):

    Lon = np.arctan2(XYZ[1],XYZ[0])

    Lat = np.arcsin(XYZ[2]/np.linalg.norm(XYZ))
    return np.rad2deg(Lat), np.rad2deg(Lon)

def radius_of_curvature(f, a=6378137, e2=0.00669438002290):
    """
    Computes the radius of curvature in the prime vertical (N).

    Parameters:
    f (float): Geodetic latitude in radians.
    a (float, optional): Semi-major axis of the reference ellipsoid. Default is GRS 80.
    e2 (float, optional): Square of the first eccentricity of the ellipsoid. Default is GRS 80.

    Returns:
    float: Radius of curvature in the prime vertical (N).
    """
    return a / np.sqrt(1 - e2 * np.sin(f)**2)

def julian_date(year, month, day, hour=0):
    """
    Computes the Julian Date for a given Gregorian date.
    
    Parameters:
    year (int): Year.
    month (int): Month.
    day (int): Day.
    hour (float, optional): Hour (fractional values allowed). Default is 0.
    
    Returns:
    float: Julian Date.
    """
    if month <= 2:
        year -= 1
        month += 12
    
    A = math.floor(year / 100)
    B = 2 - A + math.floor(A / 4)
    
    jd = (math.floor(365.25 * (year + 4716)) +
          math.floor(30.6001 * (month + 1)) + 
          day + B + hour / 24 - 1537.5)
    
    return jd

def geodetic_to_cartesian(f, l, h, a=6378137, e2=0.00669438002290):
    """
    Converts geodetic coordinates (latitude, longitude, height) to Cartesian coordinates (X, Y, Z).

    Parameters:
    f (float): Geodetic latitude in radians.
    l (float): Geodetic longitude in radians.
    h (float): Ellipsoidal height in meters.
    a (float, optional): Semi-major axis of the reference ellipsoid. Default is GRS 80.
    e2 (float, optional): Square of the first eccentricity of the ellipsoid. Default is GRS 80.

    Returns:
    tuple: (X, Y, Z) Cartesian coordinates in meters.
    """
    N = radius_of_curvature(f, a, e2)
    cos_f, sin_f = math.cos(f), math.sin(f)
    cos_l, sin_l = math.cos(l), math.sin(l)
    
    X = (N + h) * cos_f * cos_l
    Y = (N + h) * cos_f * sin_l
    Z = (N * (1 - e2) + h) * sin_f
    
    return X, Y, Z

def rotation_matrix_neu(f, l):
    """
    Computes the rotation matrix from geodetic coordinates (φ, λ) to a local topocentric (NEU) coordinate system.

    Parameters:
    f (float): Geodetic latitude in radians.
    l (float): Geodetic longitude in radians.

    Returns:
    numpy.ndarray: 3x3 rotation matrix.
    """
    sin_f, cos_f = np.sin(f), np.cos(f)
    sin_l, cos_l = np.sin(l), np.cos(l)
    
    return np.array([[-sin_f * cos_l, -sin_l, cos_f * cos_l],
                     [-sin_f * sin_l, cos_l, cos_f * sin_l],
                     [cos_f, 0, sin_f]])

def ymd_to_gps(date):
    """
    Converts a Gregorian date to GPS week and seconds of the week.

    Parameters:
    date (tuple): (year, month, day, hour, minute, second).

    Returns:
    tuple: (GPS week number, seconds of the week).
    """
    year, month, day, hour, minute, second = date
    days_since_gps_epoch = julian_date(year, month, day) - julian_date(1980, 1, 6)
    gps_week = int(days_since_gps_epoch // 7)
    day_of_week = int(days_since_gps_epoch % 7)
    seconds_of_week = day_of_week * 86400 + hour * 3600 + minute * 60 + second
    
    return gps_week, seconds_of_week
