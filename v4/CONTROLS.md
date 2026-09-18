# SEP Robot Arm 0.2.7 Controls

## Start the program

From the repository root, run:

```bash
./.venv/bin/python "v4/SEP Robot Arm 0.2.7.py"
```

## Arm controls

- `I` or `Tab`: enable inverse-kinematics mode.
- `R`: reset the arm and target.
- `F2`: open or close the settings sliders.
- `F5`: save settings and the current pose.
- `F7`: load the quick-save pose.
- `X`, `Y`, or `Z`: select the target coordinate.
- `J` / `L`: move the selected coordinate by 0.1 units.
- `I` / `K`: move the selected coordinate by 0.1 units.
- `Shift` + target movement key: fine movement by 0.025 units.
- `1`-`4`: select a joint for PID settings.

The IK solver follows the target position and can blend toward the target yaw and pitch. The `Angle weight` slider controls that blend from position-first (`0`) to stronger angle influence (`1`). The end effector is kept at or above `Y = -1`. The target Z coordinate is not clamped by the robot controller.

The base joint can rotate continuously past `180` degrees so the arm can reach behind itself. The shoulder, elbow, and wrist joint limits remain active.

## Settings sliders

Press `F2` to open the slicer-style settings panel. Scroll inside the left panel to access PID gains and limits, global and per-motor speed, per-motor acceleration, IK posture and angle weighting, target coordinates, end-effector orientation, separate `min` and `max` sliders for every motor, and panel/accent colors. Click any value box to type a value, press `Enter` to apply it, or `Escape` to cancel. The save-directory box accepts typed text or `Ctrl+V` paste. Defaults are centered evenly around zero. Select a motor with `1`-`4` when changing its PID gains.

## Camera controls

- `W` / `S`: move forward/backward.
- `A` / `D`: move left/right.
- `Space`: move up.
- `Left Ctrl`: move down.
- Mouse: look around.
- `Esc`: release the mouse cursor.
- Click outside the settings panel: recapture the mouse.
- Drag the window edge or corner to resize it. The arm stays centered in the right half and the settings panel stays anchored to the bottom-left.
- Drag the vertical divider between the two halves to resize the settings and render areas.

Settings and quick-save poses are stored in `v4/settings.json` and `v4/poses.json` when saved. To change the save location, edit `SAVE_DIRECTORY` near the top-level configuration in `SEP Robot Arm 0.2.7.py`.
