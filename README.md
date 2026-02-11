# Risk-Aware Drone Path Planning System

This project implements a risk-aware planning system for autonomous drone delivery operations. The system calculates potential casualty risks associated with in-flight failures (specifically loss of thrust) and helps evaluate trajectories to minimize harm to people on the ground if such failures occur.

The software models ballistic falls after propulsion failure, generates impact probability maps based on uncertain parameters, and computes risk scores by considering population density, sheltering factors, and crash energy at potential impact locations.

## Key Components


  ###  **Ground Map Data (GroundMap.py)**

  **Purpose:** 

*  Loads and manages geographical data for risk calculation.

  **Data layers:**
  
*  Terrain height: Elevation data
*  Sheltering factor: Protection level of environment
*  Population density: Habitants per square kilometer

  **Format:** 
  
*  16-bit PNG images with pixel values representing layer data


  ###  **Ballistic Fall Simulation (BallisticFall.py)**

  **Purpose:** 
  
*  Models drone behavior after thrust failure.


  ###  **Risk Assessment (BallisticRiskMap.py)**
  
  **Purpose:** 
  
*  Calculates casualty risk for potential drone failures.
  
  **Process:**  
1. For each trajectory sample, determine crash location and fallen altitude.
2. Select appropriate impact probability map based on heading and altitude.
3. Align impact probability map with current position.
4. Calculate risk using ground map data and impact probability.

## Usage

This project is implemented as a Python module and is intended to be used
by importing its components into your own code.

There is no standalone executable script.
