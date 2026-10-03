# Hardware Setup Guide

## Equipment List
| Equipment | Model | Purpose |
|---|---|---|
| Heating/Freezing Stage | Linkam MDSG600 | Temperature control for fluid inclusion measurement |
| Microscope | Olympus CX40M | Optical observation, 200×/500× magnification |
| Industrial Camera | Blackfly BFLY-U3-23S6C | Image capture |
| Z-axis Motor | Custom stepping motor (DTStageDriver) | Automatic focusing |
| Control PC | Windows 11, x64 | System control and image processing |

## Hardware Connections
1. **Linkam MDSG600**: Connect to PC via USB; install Linkam NEXUS software
2. **Z-axis Motor**: Connect to PC via serial port (default COM3); install DTStageDriver driver
3. **Blackfly Camera**: Connect to PC via USB3.0; install Spinnaker SDK

## Software Prerequisites
1. Install Linkam NEXUS software (provided by Linkam Scientific)
2. Install .NET Framework 4.8 (required for LinkamIpc.dll)
3. Install Python 3.11 x64
4. Install Spinnaker SDK for camera (optional: screen capture mode works without it)

## Linkam GUI Calibration
The system uses PyAutoGUI to control Linkam NEXUS software via screen coordinates. Before first run, calibrate the following positions in `config.py`:

1. **Temperature input box coordinates**: Position of the temperature setpoint input field
2. **Start heating button coordinates**: Position of the start/stop button
3. **Rate input coordinates**: Heating/cooling rate input field
4. **Screen capture region**: Coordinates of the microscope viewport in the NEXUS window

### Calibration Steps
1. Open Linkam NEXUS software and maximize the viewport
2. Read the screen coordinates of each control with a screen-capture tool or
   `pyautogui.position()` and record them
3. Update the corresponding entries in `config.py`
   (`NEXUS_POS`, `TEMP_RATE_POS`, `TEMP_LIMIT_POS`, `TEMP_START_POS`, `TEMP_STOP_POS`,
   `X_INPUT_POS`, `Y_INPUT_POS`, `ZERO_BTN_POS`, `SCREEN_REGION`)
4. Test a simple temperature ramp to confirm that the clicks land on the right fields
5. Keep the NEXUS window at a fixed position and screen resolution afterwards — all
   coordinates are absolute screen coordinates

## Serial Port Configuration
1. Check Device Manager for the COM port number of the Z-axis motor
2. Update `COM_PORT` in `config.py` (default: `3`, i.e. COM3)
3. Verify motor movement by launching `python main.py --start 1 --end 1` on a mounted
   slide and confirming that the stage moves to the target coordinate

## IPC Temperature Reading
The system uses .NET Remoting IPC to read real-time temperature from Linkam software:
- Pre-compiled `LinkamIpc.dll` is provided in `src/hardware/lib/`
- Source code and compilation instructions are in `csharp_ipc_builder/`
- Requires .NET Framework 4.8 and 64-bit Python to match the DLL architecture

## Troubleshooting
- **GUI clicks not working**: Recalibrate screen coordinates; ensure NEXUS window is in the same position
- **Temperature reading fails**: Check that Linkam NEXUS is running; verify .NET Framework 4.8 is installed
- **Z-axis motor not responding**: Check COM port number; verify driver installation; ensure motor is powered on
- **Screen capture wrong area**: Adjust capture region coordinates in `config.py`
