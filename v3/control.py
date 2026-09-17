"""Pure control and persistence helpers for the robot arm."""

import json
import math
from pathlib import Path


CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ

DEFAULT_SETTINGS = {
    "control_hz": CONTROL_HZ,
    "max_joint_speed": 90.0,
    "posture_bias": 0.08,
    "angle_weight": 0.1,
    "minimum_target_y": -1.0,
    "pid": {
        "p": [5.0, 6.0, 6.0, 4.0],
        "i": [0.0, 0.0, 0.0, 0.0],
        "d": [0.35, 0.45, 0.45, 0.30],
        "integral_limit": 10.0,
        "output_limit": 180.0,
    },
}

DEFAULT_POSES = [
    {
        "name": "home",
        "target": {"x": 0.0, "y": 1.0, "z": -9.0},
        "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "angles": [0.0, 0.0, 0.0, 0.0],
    }
]


def _copy_json(value):
    return json.loads(json.dumps(value))


def _merge_defaults(value, defaults):
    if isinstance(defaults, dict):
        result = {}
        source = value if isinstance(value, dict) else {}
        for key, default in defaults.items():
            result[key] = _merge_defaults(source.get(key), default)
        return result
    return value if value is not None else defaults


def _number(value, fallback, minimum=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(fallback)
    if minimum is not None:
        result = max(float(minimum), result)
    return result


def load_json(path, defaults):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
    except (OSError, ValueError, TypeError):
        return _copy_json(defaults)
    return _merge_defaults(loaded, defaults)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def load_settings(path):
    settings = load_json(path, DEFAULT_SETTINGS)
    settings["control_hz"] = CONTROL_HZ
    settings["max_joint_speed"] = _number(settings["max_joint_speed"], 90.0, 1.0)
    settings["posture_bias"] = _number(settings["posture_bias"], 0.35, 0.0)
    settings["minimum_target_y"] = _number(settings["minimum_target_y"], -1.0)
    for key in ("p", "i", "d"):
        values = settings["pid"][key]
        if not isinstance(values, list):
            values = DEFAULT_SETTINGS["pid"][key]
        settings["pid"][key] = [_number(item, 0.0, 0.0) for item in values[:4]]
        settings["pid"][key] += [0.0] * (4 - len(settings["pid"][key]))
    settings["pid"]["integral_limit"] = _number(
        settings["pid"]["integral_limit"], 10.0, 0.0
    )
    settings["pid"]["output_limit"] = _number(
        settings["pid"]["output_limit"], 180.0, 0.0
    )
    return settings


def load_poses(path):
    poses = load_json(path, DEFAULT_POSES)
    if not isinstance(poses, list):
        return _copy_json(DEFAULT_POSES)
    valid = []
    for pose in poses:
        try:
            target = pose["target"]
            angles = [float(value) for value in pose["angles"][:4]]
            if len(angles) != 4:
                continue
            valid.append({
                "name": str(pose["name"]),
                "target": {axis: float(target[axis]) for axis in ("x", "y", "z")},
                "orientation": {
                    axis: float(pose.get("orientation", {}).get(axis, 0.0))
                    for axis in ("roll", "pitch", "yaw")
                },
                "angles": angles,
            })
        except (KeyError, TypeError, ValueError):
            continue
    return valid or _copy_json(DEFAULT_POSES)


class PID:
    def __init__(self, kp, ki, kd, integral_limit=10.0, output_limit=180.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = abs(float(integral_limit))
        self.output_limit = abs(float(output_limit))
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = None

    def update(self, target, measured, dt=CONTROL_DT):
        if dt <= 0.0:
            return 0.0
        error = float(target) - float(measured)
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        derivative = 0.0 if self.previous_error is None else (
            error - self.previous_error
        ) / dt
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))


class FixedRateScheduler:
    def __init__(self, hz=CONTROL_HZ, max_steps=5):
        self.dt = 1.0 / float(hz)
        self.max_steps = int(max_steps)
        self.accumulator = 0.0

    def advance(self, elapsed):
        self.accumulator = min(self.accumulator + max(0.0, elapsed), self.dt * self.max_steps)
        steps = 0
        while self.accumulator + 1e-12 >= self.dt and steps < self.max_steps:
            self.accumulator -= self.dt
            steps += 1
        return steps


def approach(current, target, maximum_delta):
    delta = float(target) - float(current)
    if abs(delta) <= maximum_delta:
        return float(target)
    return float(current) + math.copysign(maximum_delta, delta)
