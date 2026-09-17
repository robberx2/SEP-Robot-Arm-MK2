import numpy as np


l1 = 5
l2 = 10
l3 = 10
l4 = 5
l5 = 2


def rot_x(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform(rotation, position):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def forward_kinematics(angles):
    matrix = np.eye(4)
    joints = [matrix[:3, 3].copy()]
    axes = [matrix[:3, 2].copy()]

    matrix = matrix @ transform(rot_z(angles[0]), [0, 0, 0])
    matrix = matrix @ transform(np.eye(3), [0, 0, l1])
    joints.append(matrix[:3, 3].copy())
    axes.append(matrix[:3, 1].copy())

    matrix = matrix @ transform(rot_y(angles[1]), [0, 0, 0])
    matrix = matrix @ transform(np.eye(3), [l2, 0, 0])
    joints.append(matrix[:3, 3].copy())
    axes.append(matrix[:3, 1].copy())

    matrix = matrix @ transform(rot_y(angles[2]), [0, 0, 0])
    matrix = matrix @ transform(np.eye(3), [l3, 0, 0])
    joints.append(matrix[:3, 3].copy())
    axes.append(matrix[:3, 1].copy())

    matrix = matrix @ transform(rot_y(angles[3]), [0, 0, 0])
    matrix = matrix @ transform(np.eye(3), [l4, 0, 0])
    joints.append(matrix[:3, 3].copy())
    axes.append(matrix[:3, 0].copy())

    matrix = matrix @ transform(rot_x(angles[4]), [0, 0, 0])
    matrix = matrix @ transform(np.eye(3), [l5, 0, 0])
    joints.append(matrix[:3, 3].copy())
    axes.append(matrix[:3, 2].copy())

    matrix = matrix @ transform(rot_z(angles[5]), [0, 0, 0])
    return matrix[:3, 3], matrix[:3, :3], np.array(joints), np.array(axes)


def rotation_matrix_from_euler(roll, pitch, yaw):
    return rot_z(np.radians(yaw)) @ rot_y(np.radians(pitch)) @ rot_x(np.radians(roll))


def rotation_error(current, target):
    rotation = target @ current.T
    return np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1]
    ]) * 0.5


def jacobian(end, joints, axes):
    matrix = np.zeros((6, 6))
    for index in range(6):
        offset = end - joints[index]
        matrix[:3, index] = np.cross(axes[index], offset)
        matrix[3:, index] = axes[index]
    return matrix


def inverse_kinematics(x, y, z, pitch, yaw, roll):
    """Return six joint angles in radians for the requested pose.

    Position uses the robot's length units. Pitch, yaw, and roll are degrees.
    """
    target = np.array([x, y, z], dtype=float)
    target_rotation = rotation_matrix_from_euler(roll, pitch, yaw)
    angles = np.radians([0, 90, 0, 0, 0, 0])
    limits = [(-180, 180), (-90, 90), (-135, 135), (-90, 90), (-180, 180), (-180, 180)]

    for _ in range(300):
        end, current_rotation, joints, axes = forward_kinematics(angles)
        position_error = target - end
        orientation_error = rotation_error(current_rotation, target_rotation)
        error = np.concatenate((position_error, orientation_error * 0.02))

        if np.linalg.norm(position_error) < 0.00000001:
            break

        matrix = jacobian(end, joints, axes)
        damping = 0.05
        damped_matrix = matrix @ matrix.T + damping**2 * np.eye(6)
        delta = matrix.T @ np.linalg.solve(damped_matrix, error)
        null_space = np.eye(6) - np.linalg.pinv(matrix) @ matrix
        delta += null_space @ (-0.05 * angles)
        step = min(0.15, 0.02 + np.linalg.norm(position_error) * 0.01)
        angles += delta * step

        for index, (low, high) in enumerate(limits):
            angles[index] = np.clip(angles[index], np.radians(low), np.radians(high))

    return angles
