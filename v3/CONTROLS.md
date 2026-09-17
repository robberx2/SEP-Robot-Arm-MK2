# SEP Robot Arm 0.2.6 Controls

## Start the program

From the repository root, run:

```bash
./.venv/bin/python "v3/SEP Robot Arm 0.2.6.py"
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

## Settings sliders

Press `F2` to adjust PID gains, maximum joint speed, posture bias, angle weight, target coordinates, and end-effector roll, pitch, and yaw from `-180` to `180` degrees.

## Camera controls

- `W` / `S`: move forward/backward.
- `A` / `D`: move left/right.
- `Space`: move up.
- `Left Ctrl`: move down.
- Mouse: look around.
- `Esc`: release the mouse cursor.
- Click outside the settings panel: recapture the mouse.

Settings and quick-save poses are stored in `v3/settings.json` and `v3/poses.json` when saved.
