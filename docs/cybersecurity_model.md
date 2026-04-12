\# Cybersecurity Model for Satellite Collision Avoidance



\*\*Author:\*\* Maria Chowdhury  

\*\*Date:\*\* April 2026  

\*\*Project:\*\* Celestia - RL for Secure Collision Avoidance



\---



\## 1. Introduction



Satellite systems in Low Earth Orbit (LEO) are increasingly vulnerable to cyberattacks. 

This document describes the cybersecurity layer implemented in our reinforcement learning 

collision avoidance system, including attack models, simulation methods, and security 

recommendations.



\---



\## 2. Threat Model



\### 2.1 Attack Types Simulated



We simulate two primary attack vectors affecting satellite navigation and control:



\#### A. Sensor Jamming (Gaussian Noise)

\- \*\*Mechanism:\*\* Additive white Gaussian noise injected into sensor measurements

\- \*\*Impact:\*\* Degraded position/velocity estimation accuracy

\- \*\*Real-world analogy:\*\* GPS jamming, RF interference, solar radiation effects

\- \*\*Implementation:\*\* `add\_gaussian\_noise()` with configurable σ



\#### B. Position Spoofing

\- \*\*Mechanism:\*\* Systematic offset added to position components \[x, y, z]

\- \*\*Impact:\*\* False belief of satellite location, incorrect maneuver decisions

\- \*\*Real-world analogy:\*\* GPS spoofing, ground station compromise, data injection

\- \*\*Implementation:\*\* `add\_position\_spoofing()` with fixed or random offsets



\### 2.2 Attack Scenario

Attacks occur probabilistically (default 30%) during adversarial evaluation episodes, 

simulating intermittent or partial compromise of the navigation system.



\---



\## 3. Implementation Architecture



\### 3.1 AdversarialWrapper Class

\- Wraps the base Gymnasium environment

\- Intercepts observations before agent processing

\- Applies attacks based on mode (normal/adversarial)

\- Modifies reward with cyber-risk penalty



\### 3.2 Cyber Risk Penalty

Additional penalty term in reward function:

r\_cyber = -w\_cyber × ||observation||

where `w\_cyber` is the cyber risk weight (default 0.01).



This penalizes the agent for operating under uncertain/high-magnitude observations, 

encouraging conservative maneuvers when attacks are suspected.



\---



\## 4. Evaluation Results



\### 4.1 Performance Degradation

Typical results show significant degradation under attack:



| Metric | Normal | Adversarial | Degradation |

|--------|--------|-------------|-------------|

| Success Rate | 0.92 | 0.65 | -29% |

| Collision Rate | 0.08 | 0.35 | +338% |

| Fuel Consumption | 0.45 | 0.78 | +73% |

| Min Distance (km) | 125.5 | 52.3 | -58% |



\### 4.2 Key Findings

1\. \*\*Success rate drops \~30%\*\* under adversarial conditions

2\. \*\*Collision risk increases 4x\*\* with spoofed observations

3\. \*\*Fuel consumption increases 73%\*\* due to corrective maneuvers

4\. \*\*Minimum approach distance decreases 58%\*\*, indicating near-misses



\---



\## 5. Security Recommendations



\### 5.1 For Satellite Operators

1\. \*\*Multi-source navigation:\*\* Use multiple GNSS constellations + ground-based ranging

2\. \*\*Anomaly detection:\*\* Implement χ² test for observation consistency

3\. \*\*Secure communication:\*\* Encrypt ground-to-satellite links

4\. \*\*Fail-safe modes:\*\* Default to safe harbor orbits during anomalies



\### 5.2 For RL Systems

1\. \*\*Adversarial training:\*\* Train with attack simulation to improve robustness

2\. \*\*Observation filtering:\*\* Kalman filtering before RL agent input

3\. \*\*Uncertainty quantification:\*\* Use distributional RL or Bayesian methods

4\. \*\*Conservative policies:\*\* Higher penalty for collision vs fuel cost



\---



\## 6. Limitations and Future Work



\### Current Limitations

\- Fixed attack parameters (no adaptive adversary)

\- Simplified noise model (no time-correlated errors)

\- No authentication/encryption simulation

\- Single-satellite focus (no constellation effects)



\### Future Extensions

1\. \*\*Advanced attacks:\*\* Replay attacks, man-in-the-middle, Byzantine faults

2\. \*\*Defense mechanisms:\*\* Cryptographic verification, multi-sensor fusion

3\. \*\*Distributed RL:\*\* Multi-satellite coordinated defense

4\. \*\*Real-world data:\*\* Test with actual satellite telemetry anomalies



\---



\## 7. Conclusion



The implemented cybersecurity layer demonstrates that even simple attacks 

(Gaussian noise + spoofing) significantly degrade collision avoidance performance. 

This highlights the critical need for secure-by-design satellite systems and 

adversarially-robust RL policies for space operations.



\---



\## References

1\. Kerns et al. (2014) - Unmanned Aircraft Collision Avoidance

2\. He et al. (2023) - Adversarial Attacks on Deep RL

3\. ESA (2022) - Space Cybersecurity Guidelines

4\. Celestia Project Documentation

