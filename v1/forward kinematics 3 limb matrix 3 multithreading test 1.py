import numpy as np


def rot_x(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rot_y(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rot_z(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def transform(rotation, position):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def forward_kinematics(angles):
    transform_matrix = np.eye(4)
    joints = []
    axes = []

    joints.append(transform_matrix[:3, 3].copy())
    axes.append(transform_matrix[:3, 2].copy())

    transform_matrix = transform_matrix @ transform(rot_z(angles[0]), [0, 0, 0])
    transform_matrix = transform_matrix @ transform(np.eye(3), [0, 0, l1])
    joints.append(transform_matrix[:3, 3].copy())
    axes.append(transform_matrix[:3, 1].copy())

    transform_matrix = transform_matrix @ transform(rot_y(angles[1]), [0, 0, 0])
    transform_matrix = transform_matrix @ transform(np.eye(3), [l2, 0, 0])
    joints.append(transform_matrix[:3, 3].copy())
    axes.append(transform_matrix[:3, 1].copy())

    transform_matrix = transform_matrix @ transform(rot_y(angles[2]), [0, 0, 0])
    transform_matrix = transform_matrix @ transform(np.eye(3), [l3, 0, 0])
    joints.append(transform_matrix[:3, 3].copy())
    axes.append(transform_matrix[:3, 1].copy())

    transform_matrix = transform_matrix @ transform(rot_y(angles[3]), [0, 0, 0])
    transform_matrix = transform_matrix @ transform(np.eye(3), [l4, 0, 0])
    joints.append(transform_matrix[:3, 3].copy())
    axes.append(transform_matrix[:3, 0].copy())

    transform_matrix = transform_matrix @ transform(rot_x(angles[4]), [0, 0, 0])
    transform_matrix = transform_matrix @ transform(np.eye(3), [l5, 0, 0])
    joints.append(transform_matrix[:3, 3].copy())
    axes.append(transform_matrix[:3, 2].copy())

    transform_matrix = transform_matrix @ transform(rot_z(angles[5]), [0, 0, 0])

    return (
        transform_matrix[:3, 3],
        transform_matrix[:3, :3],
        np.array(joints),
        np.array(axes)
    )


def rotation_matrix_from_euler(roll, pitch, yaw):
    return (
        rot_z(np.radians(yaw))
        @ rot_y(np.radians(pitch))
        @ rot_x(np.radians(roll))
    )


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
    target = np.array([x, y, z], dtype=float)
    target_rotation = rotation_matrix_from_euler(roll, pitch, yaw)
    angles = np.radians([0, 90, 0, 0, 0, 0])

    limits = [
        (-180, 180),
        (-90, 90),
        (-135, 135),
        (-90, 90),
        (-180, 180),
        (-180, 180)
    ]

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

        center = np.zeros(6)
        secondary = -0.05 * (angles - center)
        null_space = np.eye(6) - np.linalg.pinv(matrix) @ matrix
        delta += null_space @ secondary

        step = min(0.15, 0.02 + np.linalg.norm(position_error) * 0.01)
        angles += delta * step

        for index, (low, high) in enumerate(limits):
            angles[index] = np.clip(
                angles[index],
                np.radians(low),
                np.radians(high)
            )

    return angles


if __name__ == "__main__":
    target = np.array([20, 10, 10])
    target_angles = inverse_kinematics(
        target[0], target[1], target[2],
        pitch=0, yaw=0, roll=0
    )
    final_position, _, _, _ = forward_kinematics(target_angles)

    print("angles (degrees):", np.round(np.degrees(target_angles), 2))
    print("position error:", np.linalg.norm(target - final_position))
    print("position:", final_position)
    print("target:", target)
import numpy as np

l1 = 5
l2 = 10
l3 = 10
l4 = 5
def inverse_kinematics(x, y, z, pitch, yaw, roll):

    target = np.array([x, y, z], dtype=float)
    target_rotation = rotation_matrix_from_euler(
        roll,
        pitch,
        yaw
    )
    angles = np.radians([
        0,
        90,
        0,
        0,
        0,
        0
    ])

    for iteration in range(300):

        end, R, joints, axes = forward_kinematics(
            angles
        )

        position_error = target - end
    T = T @ transform(np.eye(3),[l5,0,0])
        orientation_error = rotation_error(
            R,
            target_rotation
        )

        error = np.concatenate(
            (
                position_error,
                orientation_error * 0.02
            )
        )
        np.array(axes)
        if np.linalg.norm(position_error) < 0.00000001:
            break

        J = jacobian(
            end,
            joints,
            axes
        )
    pitch,
        damping = 0.05
        @
        JJ = (
            J @ J.T
            +
            damping**2*np.eye(6)
        )
        @
        dtheta = (
            J.T
            @
            np.linalg.solve(
                JJ,
                error
            )
        )
        rot_x(np.radians(roll))
        center = np.zeros(6)
    J = np.zeros((6,6))
        secondary = (
            -0.05
            *
            (angles-center)
        )
        J[:3,i] = np.cross(
        N = (
            np.eye(6)
            -
            np.linalg.pinv(J)
            @
            J
        )

        dtheta += N @ secondary

        step = min(
            0.15,
            0.02 +
            np.linalg.norm(position_error)*0.01
        )
        integral_limit=None,
        angles += dtheta * step
    ):
        limits = [
        self.kp = kp
            (-180,180),
            (-90,90),
            (-135,135),
            (-90,90),
            (-180,180),
            (-180,180)

        ]
        self.output_limit = output_limit
        for i,(low,high) in enumerate(limits):

            angles[i] = np.clip(
                angles[i],
                np.radians(low),
                np.radians(high)
            )

    return angles
            )
            +
            self.ki * self.integral
    target = np.array([20, 10, 10])
    target_angles = inverse_kinematics(
        target[0],
        target[1],
        target[2],
        0,
        0,
        0
    )
    final_end, _, _, _ = forward_kinematics(target_angles)
    final_error = np.linalg.norm(target - final_end)
    print("Started")
    target_angles, iterations = queue.get()

    p.join()

    print("Finished")


    pid = [

        PID(
            5.0,
            0.0,
            0.35,
            integral_limit=np.radians(10),
            output_limit=np.radians(180)
        ),

        PID(
            6.0,
            0.0,
            0.45,
            integral_limit=np.radians(10),
            output_limit=np.radians(180)
        ),

        PID(
            6.0,
            0.0,
            0.45,
            integral_limit=np.radians(10),
            output_limit=np.radians(180)
        ),

        PID(
            4.0,
            0.0,
            0.3,
            integral_limit=np.radians(10),
            output_limit=np.radians(180)
        ),

        PID(
            4.0,
            0.0,
            0.3,
            integral_limit=np.radians(10),
            output_limit=np.radians(180)
        ),

        PID(
            4.0,
            0.0,
            0.3,
            integral_limit=np.radians(10),
            output_limit=np.radians(180)
        )

    ]


    current_angles = np.zeros(6)


    dt = 0.02


    print("\nMoving...")


    while True:

        reached = True


        for i in range(6):

            correction = pid[i].update(

                target_angles[i],

                current_angles[i],

                dt

            )


            max_speed = np.radians(90)


            correction = np.clip(
                correction,
                -max_speed,
                max_speed
            )


            current_angles[i] += (
                correction
                *
                dt
            )


            if abs(
                target_angles[i]
                -
                current_angles[i]
            ) > np.radians(0.5):

                reached = False



        print(
            np.round(
                np.degrees(current_angles),
                2
            )
        )


        if reached:

            print("\nmovement success")

            break


        time.sleep(dt)



    end, R, joints, axes = forward_kinematics(
        current_angles
    )


    final_end, _, _, _ = forward_kinematics(
        target_angles
    )


    final_error = np.linalg.norm(
        target - final_end
    )


    print("\n========== IK STATS ==========")


    print(
        "angles (degrees):",
        np.round(np.degrees(target_angles), 2)
    )


    print(
        "error:",
        final_error
    )


    print(
        "position:",
        final_end
    )


    print(
        "target:",
        target
    )
