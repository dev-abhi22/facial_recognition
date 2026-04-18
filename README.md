# DMS Takeover Controller

This project combines a driver monitoring system (DMS) with an FSDS takeover controller.

## What it does

- `dms_monitor.py` monitors the driver using webcam face landmarks
- calculates fatigue / distraction risk
- writes takeover state to `dms_state.json`
- enables autopilot request when risk crosses threshold

- `fsds_takeover_controller.py` reads `dms_state.json`
- watches for takeover request
- switches FSDS from manual to autonomous control
- follows `path.csv` using pure pursuit + PID speed control

## Files

- `dms_monitor.py` - driver monitoring and risk detection
- `fsds_takeover_controller.py` - FSDS takeover controller
- `dms_state.json` - shared state file between DMS and controller
- `path.csv` - path points used by controller

## How to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run both files together:
```
python dms_monitor.py  #Only in python 3.11 and below
```
```
python fsds_takeover_controller.py
```

