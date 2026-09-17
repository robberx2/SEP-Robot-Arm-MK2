# SEP Robot Arm 0.2.5 Controls

## Start the program

From the repository root, run:

```bash
./.venv/bin/python "v2/SEP Robot Arm 0.2.5.py"
```

The control dashboard appears in the upper-left corner of the window.

## Control modes

- `I`: enable inverse-kinematics position mode.
- `Tab`: return to end-effector position mode.
- `R`: reset the arm angles and target position.
- `F2`: open or close the settings sliders.
- `F5`: save settings and the current pose.
- `F7`: load the `quick-save` pose.

The arm control loop runs at a fixed 50 Hz independently of rendering. Position IK and the internal PID servos calculate and move the joints; the user controls the end-effector target rather than individual joints.

## Inverse-kinematics position mode

- `X`, `Y`, or `Z`: select the target coordinate to edit.
- `J` / `L`: move the selected target coordinate left/right by 0.1 units.
- `I` / `K`: move the selected target coordinate up/down by 0.1 units.
- `Shift` + any target key: move the selected target coordinate by 0.025 units.

The inverse-kinematics solver converts the target position into base, shoulder, elbow, and wrist angles.
Targets below the safe end-effector height of approximately `Y = -0.35` are automatically raised to keep the rectangular tool above the floor. Positive world `Y` is upward in this scene; OpenGL does not invert the world coordinate. A configurable secondary posture bias favors an arm-up/high-clearance configuration while Cartesian target error remains the primary objective.

The target orientation is controlled separately with roll, pitch, and yaw sliders. A rectangular orientation marker is drawn at the target position, and the rectangular end effector follows the selected orientation.

## Settings sliders

Press `F2` and drag the sliders to change the internal PID gains, maximum joint speed, posture bias, expanded `X`/`Y`/`Z` target ranges, and end-effector roll/pitch/yaw from `-180` to `180` degrees. The `1`-`4` keys select which internal joint PID gains are displayed. The claw is represented by a rectangular end-effector block.

Settings and the quick-save pose are human-readable JSON files at `v2/settings.json` and `v2/poses.json`. Missing or malformed files fall back to defaults instead of stopping the application.

## Camera controls

- `W`: move the camera forward.
- `S`: move the camera backward.
- `A`: move the camera left.
- `D`: move the camera right.
- `Space`: move the camera up.
- `Left Ctrl`: move the camera down.
- Move the mouse: look around.
- `Esc`: release the mouse cursor and freeze camera rotation and movement.
- Click outside the settings panel: capture the mouse and resume camera control.
- Click inside the settings panel: keep the camera frozen while using sliders.

## Dashboard information

The dashboard shows:

- The current control mode.
- The selected target axis and target coordinate.
- The selected joint.
- The current base, shoulder, elbow, and wrist angles.
