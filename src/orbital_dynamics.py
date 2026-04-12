"""
Orbital Dynamics Module for Satellite Collision Avoidance
Provides realistic orbital propagation using poliastro with J2 perturbation
Author: Dr. Raouph (Person 2) - Refactored by Celestia Team
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import requests
from datetime import datetime, timedelta
import warnings

# Poliastro imports
try:
    from poliastro.twobody import Orbit
    from poliastro.twobody.propagation import propagate
    from poliastro.bodies import Earth
    from poliastro.maneuver import Maneuver
    from poliastro.core.perturbations import J2_perturbation
    from poliastro.core.propagation import func_twobody
    from poliastro.util import time_range
    from astropy import units as u
    from astropy.time import Time
    POLIASTRO_AVAILABLE = True
except ImportError:
    POLIASTRO_AVAILABLE = False
    warnings.warn("poliastro not installed. Using simplified orbital model.")

# Celestrak TLE URL
CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"


def load_tle_from_celestrak(norad_id: int) -> List[str]:
    """
    Fetch TLE data from Celestrak for given NORAD ID.
    
    Args:
        norad_id: NORAD catalog ID (e.g., 25544 for ISS)
        
    Returns:
        List of [name, line1, line2] TLE format
        
    Raises:
        ConnectionError: If Celestrak unavailable
        ValueError: If NORAD ID not found
    """
    if not POLIASTRO_AVAILABLE:
        raise RuntimeError("poliastro required for TLE loading")
    
    params = {
        'CATNR': norad_id,
        'FORMAT': 'TLE'
    }
    
    try:
        response = requests.get(CELESTRAK_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch TLE from Celestrak: {e}")
    
    lines = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
    
    if len(lines) < 3:
        raise ValueError(f"Invalid TLE data for NORAD ID {norad_id}")
    
    # Parse TLE format: name, line1, line2
    name = lines[0] if not lines[0].startswith('1 ') else f"UNKNOWN_{norad_id}"
    line1_idx = 0 if lines[0].startswith('1 ') else 1
    
    tle = [
        name,
        lines[line1_idx],
        lines[line1_idx + 1]
    ]
    
    # Validate checksum
    if not _validate_tle_checksum(tle[1]) or not _validate_tle_checksum(tle[2]):
        warnings.warn(f"TLE checksum validation failed for {norad_id}")
    
    return tle


def _validate_tle_checksum(line: str) -> bool:
    """Validate TLE line checksum (last digit)."""
    if len(line) < 69:
        return False
    
    checksum = 0
    for char in line[:-1]:
        if char.isdigit():
            checksum += int(char)
        elif char == '-':
            checksum += 1
    
    return (checksum % 10) == int(line[-1])


def create_orbit_from_tle(tle: List[str]) -> Orbit:
    """
    Create poliastro Orbit object from TLE.
    
    Args:
        tle: [name, line1, line2] format
        
    Returns:
        poliastro.twobody.Orbit in Earth-centered inertial frame
    """
    if not POLIASTRO_AVAILABLE:
        raise RuntimeError("poliastro required for orbit creation")
    
    from poliastro.twobody import Orbit as PoliastroOrbit
    
    # Parse TLE epoch
    epoch_year = int(tle[1][18:20])
    epoch_day = float(tle[1][20:32])
    
    # Convert to full year
    if epoch_year < 57:
        full_year = 2000 + epoch_year
    else:
        full_year = 1900 + epoch_year
    
    # Create datetime from day of year
    epoch = datetime(full_year, 1, 1) + timedelta(days=epoch_day - 1)
    
    # Parse orbital elements from TLE
    # Mean motion (rev/day) -> semi-major axis
    mean_motion = float(tle[2][52:63].strip())  # revolutions per day
    
    # Eccentricity (decimal point assumed)
    eccentricity = float("0." + tle[2][26:33].strip())
    
    # Inclination (degrees)
    inclination = float(tle[2][8:16].strip()) * u.deg
    
    # RAAN (degrees)
    raan = float(tle[2][17:25].strip()) * u.deg
    
    # Argument of perigee (degrees)
    argp = float(tle[2][34:42].strip()) * u.deg
    
    # Mean anomaly (degrees)
    mean_anomaly = float(tle[2][43:51].strip()) * u.deg
    
    # Convert mean motion to semi-major axis using Kepler's 3rd law
    n = mean_motion * 2 * np.pi / 86400  # rad/s
    mu = Earth.k.to_value(u.m**3 / u.s**2)
    a = (mu / n**2)**(1/3) / 1000  # km
    
    # Create orbit
    orbit = PoliastroOrbit.from_classical(
        Earth,
        a * u.km,
        eccentricity * u.one,
        inclination,
        raan,
        argp,
        mean_anomaly,
        Time(epoch)
    )
    
    return orbit


def propagate_orbit(
    orbit: Orbit,
    duration: float,
    method: str = 'cowell',
    J2: bool = True
) -> Orbit:
    """
    Propagate orbit forward in time.
    
    Args:
        orbit: Initial poliastro Orbit
        duration: Time to propagate in seconds
        method: Propagation method ('cowell', 'kepler', 'enckle')
        J2: Include J2 perturbation (Earth oblateness)
        
    Returns:
        Propagated Orbit
    """
    if not POLIASTRO_AVAILABLE:
        raise RuntimeError("poliastro required for propagation")
    
    if method == 'cowell' and J2:
        # Use Cowell's method with J2 perturbation
        from poliastro.twobody.propagation import cowell
        
        def f(t0, u_, k):
            du_ = func_twobody(t0, u_, k)
            du_ += J2_perturbation(t0, u_, k, J2=Earth.J2.value, R=Earth.R.to_value(u.km))
            return du_
        
        propagated = cowell(orbit, duration * u.s, f=f)
    else:
        # Simple two-body propagation
        propagated = propagate(orbit, duration * u.s, method=method)
    
    return propagated


def calculate_relative_state(
    orbit_chaser: Orbit,
    orbit_threat: Orbit
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate relative position and velocity in LVLH frame.
    
    LVLH frame:
    - x: radial (nadir direction)
    - y: in-track (velocity direction)
    - z: cross-track (orbit normal)
    
    Args:
        orbit_chaser: Chaser satellite orbit
        orbit_threat: Threat (debris) orbit
        
    Returns:
        (rel_position, rel_velocity, distance) in km and km/s
    """
    if not POLIASTRO_AVAILABLE:
        raise RuntimeError("poliastro required for relative state")
    
    # Get state vectors in ECI frame
    r_chaser = orbit_chaser.r.to_value(u.km)  # Position vector
    v_chaser = orbit_chaser.v.to_value(u.km / u.s)  # Velocity vector
    
    r_threat = orbit_threat.r.to_value(u.km)
    v_threat = orbit_threat.v.to_value(u.km / u.s)
    
    # Relative state in ECI
    r_rel_eci = r_threat - r_chaser  # Threat relative to chaser
    v_rel_eci = v_threat - v_chaser
    
    # Transform to LVLH frame
    # LVLH basis vectors
    r_norm = np.linalg.norm(r_chaser)
    i_r = r_chaser / r_norm  # Radial (nadir)
    i_v = v_chaser / np.linalg.norm(v_chaser)  # Velocity direction
    i_h = np.cross(r_chaser, v_chaser)
    i_h = i_h / np.linalg.norm(i_h)  # Orbit normal (cross-track)
    i_y = np.cross(i_h, i_r)  # In-track (perpendicular to radial in orbit plane)
    i_y = i_y / np.linalg.norm(i_y)
    
    # Rotation matrix from ECI to LVLH
    R_lvlh_eci = np.vstack([i_r, i_y, i_h])
    
    # Transform relative state
    r_rel_lvlh = R_lvlh_eci @ r_rel_eci
    v_rel_lvlh = R_lvlh_eci @ v_rel_eci
    
    distance = np.linalg.norm(r_rel_lvlh)
    
    return r_rel_lvlh, v_rel_lvlh, distance


def apply_impulsive_maneuver(
    orbit: Orbit,
    delta_v: np.ndarray,
    frame: str = 'lvlh'
) -> Orbit:
    """
    Apply impulsive delta-V maneuver.
    
    Args:
        orbit: Current orbit
        delta_v: Delta-V vector [km/s]
        frame: 'lvlh' or 'eci'
        
    Returns:
        New orbit after maneuver
    """
    if not POLIASTRO_AVAILABLE:
        raise RuntimeError("poliastro required for maneuvers")
    
    # Current state
    r = orbit.r.to_value(u.km)
    v = orbit.v.to_value(u.km / u.s)
    
    if frame == 'lvlh':
        # Transform delta-V from LVLH to ECI
        r_norm = np.linalg.norm(r)
        i_r = r / r_norm
        i_v = v / np.linalg.norm(v)
        i_h = np.cross(r, v)
        i_h = i_h / np.linalg.norm(i_h)
        i_y = np.cross(i_h, i_r)
        i_y = i_y / np.linalg.norm(i_y)
        
        # LVLH to ECI rotation
        R_eci_lvlh = np.column_stack([i_r, i_y, i_h])
        delta_v_eci = R_eci_lvlh @ delta_v
    else:
        delta_v_eci = delta_v
    
    # Apply maneuver
    v_new = v + delta_v_eci
    
    # Create new orbit
    from poliastro.twobody import Orbit as PoliastroOrbit
    new_orbit = PoliastroOrbit.from_vectors(
        Earth,
        r * u.km,
        v_new * u.km / u.s,
        orbit.epoch
    )
    
    return new_orbit


def detect_close_approach(
    orbit_chaser: Orbit,
    orbit_threat: Orbit,
    time_span: float,
    time_step: float = 60.0
) -> Tuple[float, float]:
    """
    Detect time and distance of closest approach (TCA).
    
    Args:
        orbit_chaser: Chaser orbit
        orbit_threat: Threat orbit
        time_span: Search duration in seconds
        time_step: Propagation step in seconds
        
    Returns:
        (minimum_distance_km, time_to_tca_seconds)
    """
    if not POLIASTRO_AVAILABLE:
        raise RuntimeError("poliastro required for close approach detection")
    
    min_distance = float('inf')
    tca = 0.0
    
    # Propagate both orbits and find minimum distance
    for t in np.arange(0, time_span, time_step):
        t_seconds = float(t)
        
        # Propagate to time t
        orbit_c_t = propagate_orbit(orbit_chaser, t_seconds, J2=False)
        orbit_t_t = propagate_orbit(orbit_threat, t_seconds, J2=False)
        
        # Calculate distance
        r_c = orbit_c_t.r.to_value(u.km)
        r_t = orbit_t_t.r.to_value(u.km)
        distance = np.linalg.norm(r_t - r_c)
        
        if distance < min_distance:
            min_distance = distance
            tca = t_seconds
    
    # Refine with smaller step around TCA
    if tca > 0:
        for t in np.arange(max(0, tca - time_step), min(time_span, tca + time_step), time_step / 10):
            orbit_c_t = propagate_orbit(orbit_chaser, t, J2=False)
            orbit_t_t = propagate_orbit(orbit_threat, t, J2=False)
            
            r_c = orbit_c_t.r.to_value(u.km)
            r_t = orbit_t_t.r.to_value(u.km)
            distance = np.linalg.norm(r_t - r_c)
            
            if distance < min_distance:
                min_distance = distance
                tca = t
    
    return min_distance, tca


def save_orbit_plot(
    orbit: Orbit,
    filename: str,
    title: str = "Orbit",
    show_earth: bool = True
) -> None:
    """
    Save orbit visualization to file.
    
    Args:
        orbit: Orbit to plot
        filename: Output file path (PDF/PNG)
        title: Plot title
        show_earth: Show Earth sphere
    """
    if not POLIASTRO_AVAILABLE:
        raise RuntimeError("poliastro required for plotting")
    
    from poliastro.plotting import StaticOrbitPlotter
    
    plotter = StaticOrbitPlotter()
    plotter.plot(orbit, label=title)
    
    if show_earth:
        from poliastro.bodies import Earth as EarthBody
        plotter.plot_body(EarthBody)
    
    plt = plotter.generate()
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Orbit plot saved to {filename}")


def create_scenario_json(
    chaser_norad: int,
    threat_norad: int,
    output_file: str
) -> Dict:
    """
    Create scenario file from NORAD IDs.
    
    Args:
        chaser_norad: Chaser satellite NORAD ID
        threat_norad: Threat satellite NORAD ID
        output_file: Output JSON file path
        
    Returns:
        Scenario dictionary
    """
    import json
    
    # Load TLEs
    tle_chaser = load_tle_from_celestrak(chaser_norad)
    tle_threat = load_tle_from_celestrak(threat_norad)
    
    # Create orbits and find TCA
    orbit_c = create_orbit_from_tle(tle_chaser)
    orbit_t = create_orbit_from_tle(tle_threat)
    
    min_dist, tca = detect_close_approach(orbit_c, orbit_t, 24 * 3600)
    
    scenario = {
        'name': f'Conjunction_{chaser_norad}_{threat_norad}',
        'chaser_norad_id': chaser_norad,
        'threat_norad_id': threat_norad,
        'chaser_tle': tle_chaser,
        'threat_tle': tle_threat,
        'tca_seconds': tca,
        'min_distance_km': min_dist,
        'created': datetime.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(scenario, f, indent=2)
    
    print(f"Scenario saved to {output_file}")
    print(f"TCA: {tca/60:.1f} minutes, Min distance: {min_dist:.3f} km")
    
    return scenario


# Simplified fallback for testing without poliastro
class SimplifiedOrbit:
    """Fallback orbit model when poliastro unavailable."""
    
    def __init__(self, position: np.ndarray, velocity: np.ndarray, epoch: datetime):
        self.r = position * u.km if POLIASTRO_AVAILABLE else position
        self.v = velocity * u.km / u.s if POLIASTRO_AVAILABLE else velocity
        self.epoch = epoch


if __name__ == "__main__":
    print("=" * 70)
    print("Orbital Dynamics Module - Test Suite")
    print("=" * 70)
    
    if not POLIASTRO_AVAILABLE:
        print("ERROR: poliastro not installed. Install with:")
        print("  pip install poliastro astropy")
        exit(1)
    
    # Test 1: Load TLE
    print("\n[Test 1] Loading TLE from Celestrak...")
    try:
        tle_iss = load_tle_from_celestrak(25544)
        print(f"  ✓ Loaded: {tle_iss[0]}")
        print(f"  Line 1: {tle_iss[1][:50]}...")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 2: Create orbit
    print("\n[Test 2] Creating orbit from TLE...")
    try:
        orbit_iss = create_orbit_from_tle(tle_iss)
        print(f"  ✓ Orbit created")
        print(f"  Semi-major axis: {orbit_iss.a:.1f}")
        print(f"  Eccentricity: {orbit_iss.ecc:.6f}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 3: Propagation
    print("\n[Test 3] Propagating orbit...")
    try:
        orbit_future = propagate_orbit(orbit_iss, 3600)  # 1 hour
        print(f"  ✓ Propagated 1 hour")
        print(f"  Position change: {np.linalg.norm(orbit_future.r - orbit_iss.r):.1f} km")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 4: Relative state
    print("\n[Test 4] Calculating relative state...")
    try:
        # Create a nearby orbit
        tle_debris = load_tle_from_celestrak(12345)  # Example debris
        orbit_debris = create_orbit_from_tle(tle_debris)
        
        rel_pos, rel_vel, distance = calculate_relative_state(orbit_iss, orbit_debris)
        print(f"  ✓ Relative state calculated")
        print(f"  Distance: {distance:.3f} km")
        print(f"  Relative velocity: {np.linalg.norm(rel_vel):.3f} km/s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 5: Close approach detection
    print("\n[Test 5] Detecting close approach...")
    try:
        min_dist, tca = detect_close_approach(orbit_iss, orbit_debris, 24*3600)
        print(f"  ✓ Close approach detected")
        print(f"  TCA: {tca/60:.1f} minutes")
        print(f"  Min distance: {min_dist:.3f} km")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 6: Maneuver
    print("\n[Test 6] Applying impulsive maneuver...")
    try:
        delta_v = np.array([0.01, 0, 0])  # 10 m/s radial
        orbit_maneuvered = apply_impulsive_maneuver(orbit_iss, delta_v)
        print(f"  ✓ Maneuver applied")
        print(f"  Delta-V: {np.linalg.norm(delta_v)*1000:.1f} m/s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)