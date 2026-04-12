from environment import SatelliteCAMEnv
import numpy as np

def test_basic():
    """Test basic environment functionality"""
    print("Testing SatelliteCAMEnv...")
    
    env = SatelliteCAMEnv(use_real_orbital=False, max_steps=100)
    obs, info = env.reset(seed=42)
    
    assert obs.shape == (7,)
    assert 'initial_distance' in info
    
    print(f"✓ Reset successful: initial_distance={info['initial_distance']:.3f} km")
    
    total_reward = 0
    step_count = 0
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        assert obs.shape == (7,)
        
        if terminated or truncated:
            break
    
    print(f"✓ Episode completed: {step_count} steps, total_reward={total_reward:.2f}")
    print(f"✓ Termination: {info.get('termination_reason')}, Success: {info.get('success')}")
    
    print("\nAll tests passed! ✓")

if __name__ == "__main__":
    test_basic()