# Satellite Collision Avoidance Simulation

This project simulates the orbits of multiple satellites and their interactions in a Low Earth Orbit (LEO) environment. It is designed to test scenarios where satellites may need to perform unexpected maneuvers due to potential collisions, prompting nearby satellites to take safety actions to maintain a safe distance.

## Overview

The simulation demonstrates how satellites in a constellation might autonomously react to unexpected maneuvers by one satellite, ensuring that collisions are avoided. This project lays the groundwork for testing how multi-agent deep reinforcement learning can be applied to develop an optimized autonomous maneuvering system for satellites.

## Features

- **Simulates orbital mechanics** for multiple satellites in different orbital planes.
- **Collision avoidance**: Satellites check the proximity to their neighbors and flag a warning if they get too close.
- **Unexpected maneuvers**: Randomly selected satellites may change their speed, forcing others to react.
- **Visualization**: A 3D visualization of the satellite orbits and their interactions is provided using Matplotlib.

## Simulation Example

The example below shows how satellites avoid collisions during unexpected maneuvers:

![Satellite Simulation](satellite_simulation.gif)

## Contact me
If you encounter any issues with the reproducibility of this simulator or would like to learn more about my research, please feel free to visit my [Google Scholar profile](https://scholar.google.es/citations?user=6PZm2aYAAAAJ&hl=es&oi=ao) or contact me directly via email at flozano@ic.uma.es.

