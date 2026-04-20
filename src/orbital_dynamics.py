"""
Orbital Dynamics Module
Provides TLE handling, orbit propagation, and relative state calculations
Author: Dr. Raouph (Converted from Jupyter notebook by Celestia Team)
"""

import requests
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass

# Poliastro imports for orbital mechanics
try:
    from sgp4.api import Satrec
    from astropy import units as u
    from astropy.time import Time, TimeDelta
    from poliastro.twobody.orbit import Orbit
    from poliastro.bodies import Earth
    POLIASTRO_AVAILABLE = True
except ImportError:
    POLIASTRO_AVAILABLE = False
    print("Warning: poliastro/sgp4 not available, using fallback mode")

@dataclass
class TLEData:
    """Structure to hold TLE data"""
    name: str
    line1: str
    line2: str
    norad_id: int

def load_tle_from_celestrak(norad_id: Optional[int] = None, group: str = "active") -> Union[TLEData, List[TLEData]]:
    """
    Load TLE data from Celestrak.org
    
    Args:
        norad_id: Specific NORAD ID to fetch (if None, fetches group)
        group: Group name (e.g., 'active', 'iridium-33-debris', 'stations')
        
    Returns:
        TLEData object or list of TLEData objects
    """
    if not POLIASTRO_AVAILABLE:
        raise ImportError("poliastro and sgp4 required for TLE processing")
    
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
    
    print(f"Fetching TLE data from: {url}")
    response = requests.get(url, timeout=30)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch TLE data: HTTP {response.status_code}")
    
    lines = response.text.strip().split("\n")
    tles = []
    
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            name = lines[i].strip()
            line1 = lines[i+1].strip()
            line2 = lines[i+2].strip()
            
            # Extract NORAD ID from line1 (characters 2-7)
            try:
                norad = int(line1[2:7])
            except:
                norad = 0
                
            tles.append(TLEData(name=name, line1=line1, line2=line2, norad_id=norad))
    
    if norad_id is not None:
        for tle in tles:
            if tle.norad_id == norad_id:
                return tle
        raise ValueError(f"NORAD ID {norad_id} not found in group {group}")
    
    return tles

def load_tle_from_file(filepath: str) -> Tuple[TLEData, TLEData]:
    """
    Parse a TLE file containing two satellite objects.
    Returns: (chaser_tle, threat_tle)
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    tles = []
    i = 0
    
    while i < len(lines):
        # Look for line 1 (starts with '1 ')
        if lines[i].startswith('1 ') and i >= 1:
            name = lines[i-1]
            line1 = lines[i]
            line2 = lines[i+1] if i+1 < len(lines) else ""
            
            if line2.startswith('2 '):
                try:
                    norad = int(line1[2:7])
                except:
                    norad = 0
                tles.append(TLEData(name=name, line1=line1, line2=line2, norad_id=norad))
                i += 2
        i += 1
    
    if len(tles) < 2:
        raise ValueError(f"Expected 2 TLE objects, found {len(tles)} in {filepath}")
    
    return tles[0], tles[1]

def create_orbit_from_tle(tle_data: Union[TLEData, List[str], Dict]) -> Orbit:
    """
    Create poliastro Orbit object from TLE data
    
    Args:
        tle_data: TLEData object, or list [name, line1, line2], or dict with keys
        
    Returns:
        poliastro.twobody.orbit.Orbit
    """
    if not POLIASTRO_AVAILABLE:
        raise ImportError("poliastro required for orbit creation")
    
    # Handle different input types
    if isinstance(tle_data, TLEData):
        line1, line2 = tle_data.line1, tle_data.line2
    elif isinstance(tle_data, (list, tuple)):
        if len(tle_data) >= 3:
            line1, line2 = tle_data[1], tle_data[2]
        else:
            raise ValueError("TLE list must have at least 3 elements [name, line1, line2]")
    elif isinstance(tle_data, dict):
        line1 = tle_data.get('line1') or tle_data.get('line1_deb')
        line2 = tle_data.get('line2') or tle_data.get('line2_deb')
    else:
        raise ValueError(f"Unsupported TLE data type: {type(tle_data)}")
    
    # Use SGP4 to get initial state vectors
    sat = Satrec.twoline2rv(line1, line2)
    
    # Get epoch from Satrec
    epoch_jd = sat.jdsatepoch
    
    # Propagate to epoch (0 minutes from epoch)
    error, r, v = sat.sgp4(epoch_jd, 0.0)
    
    if error != 0:
        raise Exception(f"SGP4 propagation error: code {error}")
    
    # Convert to astropy units
    r = np.array(r) * u.km
    v = np.array(v) * u.km / u.s
    
    # Create poliastro Orbit
    orbit = Orbit.from_vectors(Earth, r, v, epoch=Time(epoch_jd, format='jd'))
    
    return orbit

def propagate_orbit(orbit: Orbit, time_seconds: float) -> Orbit:
    """
    Propagate orbit forward by specified time
    
    Args:
        orbit: Initial poliastro orbit
        time_seconds: Time to propagate in seconds
        
    Returns:
        Propagated orbit
    """
    if not POLIASTRO_AVAILABLE:
        raise ImportError("poliastro required for propagation")
    
    # Convert seconds to time delta
    dt = TimeDelta(time_seconds * u.s)
    
    # Propagate
    new_orbit = orbit.propagate(dt)
    
    return new_orbit

def calculate_relative_state(orbit_chaser: Orbit, orbit_threat: Orbit) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate relative state between chaser and threat satellites
    
    Args:
        orbit_chaser: Orbit of chaser (active satellite)
        orbit_threat: Orbit of threat (debris)
        
    Returns:
        (rel_position, rel_velocity, distance)
        - rel_position: numpy array [x, y, z] in km (LVLH frame approximation)
        - rel_velocity: numpy array [vx, vy, vz] in km/s
        - distance: scalar distance in km
    """
    if not POLIASTRO_AVAILABLE:
        raise ImportError("poliastro required for state calculation")
    
    # Get position and velocity vectors
    r_chaser = orbit_chaser.r.to(u.km).value
    v_chaser = orbit_chaser.v.to(u.km / u.s).value
    
    r_threat = orbit_threat.r.to(u.km).value
    v_threat = orbit_threat.v.to(u.km / u.s).value
    
    # Calculate relative position and velocity
    rel_pos = np.array(r_chaser) - np.array(r_threat)
    rel_vel = np.array(v_chaser) - np.array(v_threat)
    
    # Calculate distance
    distance = np.linalg.norm(rel_pos)
    
    return rel_pos, rel_vel, float(distance)

def apply_impulsive_maneuver(orbit: Orbit, delta_v: np.ndarray) -> Orbit:
    """
    Apply impulsive maneuver (delta-V) to orbit
    
    Args:
        orbit: Current orbit
        delta_v: Delta-V vector [vx, vy, vz] in km/s
        
    Returns:
        New orbit after maneuver
    """
    if not POLIASTRO_AVAILABLE:
        raise ImportError("poliastro required for maneuver")
    
    # Get current state
    r = orbit.r
    v = orbit.v
    
    # Apply delta-V (convert to same units as v)
    delta_v_astropy = np.array(delta_v) * u.km / u.s
    v_new = v + delta_v_astropy
    
    # Create new orbit from vectors
    new_orbit = Orbit.from_vectors(Earth, r, v_new, epoch=orbit.epoch)
    
    return new_orbit

def detect_close_approach(orbit1: Orbit, orbit2: Orbit, duration_seconds: float, n_steps: int = 1000) -> Tuple[float, float]:
    """
    Detect minimum distance and time of close approach between two orbits
    
    Args:
        orbit1: First orbit
        orbit2: Second orbit
        duration_seconds: Propagation duration in seconds
        n_steps: Number of propagation steps
        
    Returns:
        (minimum_distance_km, time_of_close_approach_seconds)
    """
    if not POLIASTRO_AVAILABLE:
        raise ImportError("poliastro required for close approach detection")
    
    min_distance = float('inf')
    tca = 0.0
    
    # Generate time steps
    times = np.linspace(0, duration_seconds, n_steps)
    
    for t in times:
        # Propagate both orbits
        dt = TimeDelta(t * u.s)
        orb1_prop = orbit1.propagate(dt)
        orb2_prop = orbit2.propagate(dt)
        
        # Calculate distance
        r1 = orb1_prop.r.to(u.km).value
        r2 = orb2_prop.r.to(u.km).value
        dist = np.linalg.norm(np.array(r1) - np.array(r2))
        
        if dist < min_distance:
            min_distance = dist
            tca = t
    
    return float(min_distance), float(tca)

def generate_scenario_file(filename: str, chaser_norad: int, threat_norad: int, 
                          scenario_name: str = "Collision Scenario"):
    """
    Generate a JSON scenario file for specific satellite pair
    """
    import json
    
    scenario = {
        "name": scenario_name,
        "chaser_norad_id": chaser_norad,
        "threat_norad_id": threat_norad,
        "time_step_seconds": 60,
        "collision_distance_km": 0.1,
        "safe_distance_km": 5.0,
        "max_simulation_time_hours": 24
    }
    
    with open(filename, 'w') as f:
        json.dump(scenario, f, indent=2)
    
    print(f"Scenario saved to {filename}")

# Example usage and testing
if __name__ == "__main__":
    print("Testing orbital_dynamics module...")
    
    if POLIASTRO_AVAILABLE:
        print("✓ Poliastro available")
        
        # Test with default scenario (Iridium 33)
        try:
            print("\nFetching TLE for Iridium 33 (NORAD 24946)...")
            tle = load_tle_from_celestrak(norad_id=24946, group="iridium-33-debris")
            print(f"Found: {tle.name}")
            
            orbit = create_orbit_from_tle(tle)
            print(f"Orbit created: Epoch {orbit.epoch}")
            print(f"Position: {orbit.r}")
            print(f"Velocity: {orbit.v}")
            
            # Test propagation
            future_orbit = propagate_orbit(orbit, 3600)  # 1 hour
            print(f"Propagated 1 hour successfully")
            
        except Exception as e:
            print(f"Error in test: {e}")
    else:
        print("✗ Poliastro not available - install with: pip install poliastro sgp4 astropy")