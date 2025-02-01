# behavioral_data_visualization


## Features  
✅ Extracts and processes behavioral data from BMAD sensor recordings  
✅ Synchronizes raw video with sensor data timestamps 
✅ Generates final processed video for research analysis  

---

### `makeRollingData_new.py`  
- **Generates rolling behavior probability plots from sensor data**
  - Reads BMAD accelerometer data 
  - Applies time padding and smoothing for visualization  
  - Creates real-time probability plots of behaviors
  - Exports animated behavior data** for use

### `combineVids.py`  
- **Combines raw behavior footage with animated behavioral visualizations**
  - Reads behavior predictions from CSV file  
  - Extracts relevant time-aligned video frames  
  - Overlays behavior labels (both manual and predicted) onto video frames  
  - Merges original footage with animated graphs for analysis   
