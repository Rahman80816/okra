from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from arm_interpolator import ArmInterpolator


DEFAULT_SCENE_XML = "unitree_mujoco/unitree_robots/g1/scene_29dof_with_hand.xml"
DEFAULT_POSE_JSON = Path(__file__).resolve().parent / "data" / "mujoco_right_arm_drop_pose.json"

SIMULATE_DT = 0.005
VIEWER_DT = 0.02
WAIT_BEFORE_MOVE_S = 3.0
MIN_MOVE_DURATION_S = 1.5
MAX_JOINT_SPEED_RAD_S = 0.5
GRIPPER_OPEN_DURATION_S = 0.8
HOLD_AFTER_DONE_S = 1.0
FREE_BASE_QPOS = slice(0, 7)
FREE_BASE_QVEL = slice(0, 6)
SAFE_START_MAX_TRIES = 200
DROP_HEIGHT_ABOVE_LEFT_HAND_M = 0.24
LEFT_BASKET_TARGET_REL_PELVIS = np.array([0.30, 0.08, -0.07])
RIGHT_DROP_ELBOW_TARGET_REL_PELVIS = np.array([0.18, -0.18, 0.10])
RIGHT_DROP_ELBOW_TASK_WEIGHT = 0.7
RIGHT_DROP_MIN_ELBOW_X_REL_PELVIS = 0.10
RIGHT_HARVEST_TARGET_X_RANGE = (0.36, 0.58)
RIGHT_HARVEST_TARGET_Y_RANGE = (-0.38, 0.10)
RIGHT_HARVEST_TARGET_Z_RANGE = (-0.10, 0.20)
RIGHT_HARVEST_MAX_ELBOW_RAD = 1.05
RIGHT_HARVEST_MAX_IK_ERROR_M = 0.035
IK_MAX_ITERS = 300
IK_POS_TOLERANCE_M = 0.01
IK_DAMPING = 0.08
IK_MAX_STEP_RAD = 0.05

RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

RIGHT_HAND_JOINTS = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

RIGHT_HAND_TARGET_BODY = "right_hand_index_0_link"
LEFT_HAND_TARGET_BODY = "left_hand_index_0_link"
RIGHT_ELBOW_TARGET_BODY = "right_elbow_link"

RIGHT_DROP_POSE_INITIAL_GUESS = {
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.3,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

RIGHT_HARVEST_POSE_INITIAL_GUESS = {
    "right_shoulder_pitch_joint": -0.25,
    "right_shoulder_roll_joint": 0.25,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.35,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

LEFT_BASKET_POSE_INITIAL_GUESS = {
    "left_shoulder_pitch_joint": -0.1,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.1,
    "left_elbow_joint": 0.45,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
}

FALLBACK_FRONT_START_POSE = {
    "right_shoulder_pitch_joint": -0.25,
    "right_shoulder_roll_joint": 0.25,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.35,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

RIGHT_HAND_CLOSED_POSE = {
    "right_hand_thumb_0_joint": 0.35,
    "right_hand_thumb_1_joint": 0.35,
    "right_hand_thumb_2_joint": -0.9,
    "right_hand_index_0_joint": 1.0,
    "right_hand_index_1_joint": 1.1,
    "right_hand_middle_0_joint": 1.0,
    "right_hand_middle_1_joint": 1.1,
}

RIGHT_HAND_OPEN_POSE = {
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": 0.0,
    "right_hand_thumb_2_joint": 0.0,
    "right_hand_index_0_joint": 0.0,
    "right_hand_index_1_joint": 0.0,
    "right_hand_middle_0_joint": 0.0,
    "right_hand_middle_1_joint": 0.0,
}


@dataclass(frozen=True)
class JointActuator:
    joint_name: str
    actuator_name: str
    joint_id: int
    actuator_id: int
    qposadr: int
    qveladr: int
    joint_range: tuple[float, float]
    ctrlrange: tuple[float, float]


def actuator_name_for_joint(joint_name: str) -> str:
    return joint_name.removesuffix("_joint")


def get_joint_actuator(model: mujoco.MjModel, joint_name: str) -> JointActuator:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"joint not found: {joint_name}")

    actuator_name = actuator_name_for_joint(joint_name)
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
    )
    if actuator_id < 0:
        raise ValueError(f"actuator not found: {actuator_name}")

    joint_range = tuple(float(v) for v in model.jnt_range[joint_id])
    ctrlrange = tuple(float(v) for v in model.actuator_ctrlrange[actuator_id])
    return JointActuator(
        joint_name=joint_name,
        actuator_name=actuator_name,
        joint_id=joint_id,
        actuator_id=actuator_id,
        qposadr=int(model.jnt_qposadr[joint_id]),
        qveladr=int(model.jnt_dofadr[joint_id]),
        joint_range=joint_range,
        ctrlrange=ctrlrange,
    )


def get_actuated_joint_names(model: mujoco.MjModel) -> list[str]:
    joint_names = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id][0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is not None:
            joint_names.append(joint_name)
    return joint_names


def read_targets(
    data: mujoco.MjData, refs: dict[str, JointActuator], joint_names: tuple[str, ...] | list[str]
) -> dict[str, float]:
    return {name: float(data.qpos[refs[name].qposadr]) for name in joint_names}


def clamp_targets(
    targets: dict[str, float], refs: dict[str, JointActuator]
) -> dict[str, float]:
    return {
        name: float(np.clip(value, refs[name].joint_range[0], refs[name].joint_range[1]))
        for name, value in targets.items()
    }


def sample_right_arm_start(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    refs: dict[str, JointActuator],
) -> dict[str, float]:
    original_qpos = data.qpos.copy()
    original_qvel = data.qvel.copy()
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("body not found: pelvis")
    try:
        for _ in range(SAFE_START_MAX_TRIES):
            mujoco.mj_forward(model, data)
            pelvis_pos = data.xpos[pelvis_id]
            target_pos = pelvis_pos + np.array(
                [
                    rng.uniform(*RIGHT_HARVEST_TARGET_X_RANGE),
                    rng.uniform(*RIGHT_HARVEST_TARGET_Y_RANGE),
                    rng.uniform(*RIGHT_HARVEST_TARGET_Z_RANGE),
                ]
            )
            start, _, ik_error_m = solve_arm_to_body_position(
                model,
                data,
                refs,
                RIGHT_ARM_JOINTS,
                RIGHT_HAND_TARGET_BODY,
                target_pos,
                clamp_targets(RIGHT_HARVEST_POSE_INITIAL_GUESS, refs),
            )
            if (
                ik_error_m > RIGHT_HARVEST_MAX_IK_ERROR_M
                or start["right_elbow_joint"] > RIGHT_HARVEST_MAX_ELBOW_RAD
            ):
                continue
            set_joint_positions(data, refs, start)
            mujoco.mj_forward(model, data)
            if is_safe_front_start(model, data):
                return start
        return dict(FALLBACK_FRONT_START_POSE)
    finally:
        data.qpos[:] = original_qpos
        data.qvel[:] = original_qvel
        mujoco.mj_forward(model, data)


def is_safe_front_start(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_0_link")
    pelvis_pos = data.xpos[pelvis_id]
    wrist_rel = data.xpos[wrist_id] - pelvis_pos
    hand_rel = data.xpos[hand_id] - pelvis_pos
    return (
        wrist_rel[0] >= 0.0
        and hand_rel[0] >= 0.05
        and -0.34 <= hand_rel[1] <= 0.22
        and hand_rel[2] >= -0.30
    )


def estimate_move_duration_s(
    start_angles: dict[str, float],
    goal_angles: dict[str, float],
    max_joint_speed_rad_s: float = MAX_JOINT_SPEED_RAD_S,
    min_duration_s: float = MIN_MOVE_DURATION_S,
) -> float:
    if max_joint_speed_rad_s <= 0:
        raise ValueError("max_joint_speed_rad_s must be positive")
    max_delta = max(abs(goal_angles[name] - start_angles[name]) for name in start_angles)
    return max(min_duration_s, max_delta / max_joint_speed_rad_s)


def solve_arm_to_body_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    refs: dict[str, JointActuator],
    joint_names: tuple[str, ...],
    body_name: str,
    target_pos: np.ndarray,
    initial_guess: dict[str, float],
) -> tuple[dict[str, float], np.ndarray, float]:
    original_qpos = data.qpos.copy()
    original_qvel = data.qvel.copy()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"body not found: {body_name}")

    try:
        set_joint_positions(data, refs, initial_guess)
        mujoco.mj_forward(model, data)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        dof_ids = np.array([refs[name].qveladr for name in joint_names], dtype=int)
        qpos_ids = [refs[name].qposadr for name in joint_names]

        for _ in range(IK_MAX_ITERS):
            mujoco.mj_forward(model, data)
            err = target_pos - data.xpos[body_id]
            if np.linalg.norm(err) <= IK_POS_TOLERANCE_M:
                break

            mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
            jac = jacp[:, dof_ids]
            lhs = jac @ jac.T + (IK_DAMPING**2) * np.eye(3)
            step = jac.T @ np.linalg.solve(lhs, err)
            step = np.clip(step, -IK_MAX_STEP_RAD, IK_MAX_STEP_RAD)

            for qpos_id, joint_name, delta in zip(qpos_ids, joint_names, step):
                lower, upper = refs[joint_name].joint_range
                data.qpos[qpos_id] = np.clip(data.qpos[qpos_id] + delta, lower, upper)

        mujoco.mj_forward(model, data)
        solved = read_targets(data, refs, joint_names)
        final_pos = data.xpos[body_id].copy()
        error_m = float(np.linalg.norm(target_pos - final_pos))
        return solved, final_pos, error_m
    finally:
        data.qpos[:] = original_qpos
        data.qvel[:] = original_qvel
        mujoco.mj_forward(model, data)


def solve_arm_to_body_position_with_secondary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    refs: dict[str, JointActuator],
    joint_names: tuple[str, ...],
    primary_body_name: str,
    primary_target_pos: np.ndarray,
    secondary_body_name: str,
    secondary_target_pos: np.ndarray,
    secondary_weight: float,
    initial_guess: dict[str, float],
) -> tuple[dict[str, float], np.ndarray, np.ndarray, float, float]:
    original_qpos = data.qpos.copy()
    original_qvel = data.qvel.copy()
    primary_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, primary_body_name
    )
    secondary_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, secondary_body_name
    )
    if primary_body_id < 0:
        raise ValueError(f"body not found: {primary_body_name}")
    if secondary_body_id < 0:
        raise ValueError(f"body not found: {secondary_body_name}")

    try:
        set_joint_positions(data, refs, initial_guess)
        mujoco.mj_forward(model, data)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        dof_ids = np.array([refs[name].qveladr for name in joint_names], dtype=int)
        qpos_ids = [refs[name].qposadr for name in joint_names]

        for _ in range(IK_MAX_ITERS):
            mujoco.mj_forward(model, data)
            primary_err = primary_target_pos - data.xpos[primary_body_id]
            secondary_err = secondary_target_pos - data.xpos[secondary_body_id]
            if (
                np.linalg.norm(primary_err) <= IK_POS_TOLERANCE_M
                and data.xpos[secondary_body_id][0] >= RIGHT_DROP_MIN_ELBOW_X_REL_PELVIS
            ):
                break

            mujoco.mj_jacBody(model, data, jacp, jacr, primary_body_id)
            primary_jac = jacp[:, dof_ids]
            mujoco.mj_jacBody(model, data, jacp, jacr, secondary_body_id)
            secondary_jac = jacp[:, dof_ids]
            jac = np.vstack((primary_jac, secondary_jac * secondary_weight))
            err = np.concatenate((primary_err, secondary_err * secondary_weight))
            lhs = jac @ jac.T + (IK_DAMPING**2) * np.eye(jac.shape[0])
            step = jac.T @ np.linalg.solve(lhs, err)
            step = np.clip(step, -IK_MAX_STEP_RAD, IK_MAX_STEP_RAD)

            for qpos_id, joint_name, delta in zip(qpos_ids, joint_names, step):
                lower, upper = refs[joint_name].joint_range
                data.qpos[qpos_id] = np.clip(data.qpos[qpos_id] + delta, lower, upper)

        mujoco.mj_forward(model, data)
        solved = read_targets(data, refs, joint_names)
        primary_pos = data.xpos[primary_body_id].copy()
        secondary_pos = data.xpos[secondary_body_id].copy()
        primary_error_m = float(np.linalg.norm(primary_target_pos - primary_pos))
        secondary_error_m = float(np.linalg.norm(secondary_target_pos - secondary_pos))
        return solved, primary_pos, secondary_pos, primary_error_m, secondary_error_m
    finally:
        data.qpos[:] = original_qpos
        data.qvel[:] = original_qvel
        mujoco.mj_forward(model, data)


LEFT_BASKET_WRIST_UP = {
    "left_wrist_roll_joint":  1.000,
    "left_wrist_pitch_joint": 0.786,
    "left_wrist_yaw_joint":   0.786,
}


def solve_left_basket_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    refs: dict[str, JointActuator],
) -> tuple[dict[str, float], np.ndarray, np.ndarray, float]:
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("body not found: pelvis")
    mujoco.mj_forward(model, data)
    target_pos = data.xpos[pelvis_id] + LEFT_BASKET_TARGET_REL_PELVIS
    solved, final_pos, error_m = solve_arm_to_body_position(
        model,
        data,
        refs,
        LEFT_ARM_JOINTS,
        LEFT_HAND_TARGET_BODY,
        target_pos,
        clamp_targets(LEFT_BASKET_POSE_INITIAL_GUESS, refs),
    )
    solved.update(LEFT_BASKET_WRIST_UP)
    return solved, target_pos, final_pos, error_m


def solve_right_arm_drop_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    refs: dict[str, JointActuator],
    left_basket_pose: dict[str, float],
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, float, float]:
    original_qpos = data.qpos.copy()
    original_qvel = data.qvel.copy()
    left_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_HAND_TARGET_BODY)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if left_body_id < 0:
        raise ValueError(f"body not found: {LEFT_HAND_TARGET_BODY}")
    if pelvis_id < 0:
        raise ValueError("body not found: pelvis")

    try:
        set_joint_positions(data, refs, left_basket_pose)
        mujoco.mj_forward(model, data)
        left_hand_pos = data.xpos[left_body_id].copy()
        target_pos = left_hand_pos + np.array([0.0, 0.0, DROP_HEIGHT_ABOVE_LEFT_HAND_M])
        elbow_target_pos = data.xpos[pelvis_id] + RIGHT_DROP_ELBOW_TARGET_REL_PELVIS
        solved, final_pos, elbow_pos, error_m, elbow_error_m = (
            solve_arm_to_body_position_with_secondary(
                model,
                data,
                refs,
                RIGHT_ARM_JOINTS,
                RIGHT_HAND_TARGET_BODY,
                target_pos,
                RIGHT_ELBOW_TARGET_BODY,
                elbow_target_pos,
                RIGHT_DROP_ELBOW_TASK_WEIGHT,
                clamp_targets(RIGHT_DROP_POSE_INITIAL_GUESS, refs),
            )
        )
        return solved, left_hand_pos, target_pos, elbow_pos, error_m, elbow_error_m
    finally:
        data.qpos[:] = original_qpos
        data.qvel[:] = original_qvel
        mujoco.mj_forward(model, data)


def set_joint_positions(
    data: mujoco.MjData, refs: dict[str, JointActuator], targets: dict[str, float]
) -> None:
    for name, value in targets.items():
        data.qpos[refs[name].qposadr] = value
        data.qvel[refs[name].qveladr] = 0.0


def apply_pd_control(
    data: mujoco.MjData,
    refs: dict[str, JointActuator],
    targets: dict[str, float],
) -> None:
    for name, target in targets.items():
        ref = refs[name]
        q = float(data.qpos[ref.qposadr])
        dq = float(data.qvel[ref.qveladr])
        kp, kd = gains_for(ref)
        torque = kp * (target - q) - kd * dq
        data.ctrl[ref.actuator_id] = np.clip(torque, ref.ctrlrange[0], ref.ctrlrange[1])


def gains_for(ref: JointActuator) -> tuple[float, float]:
    if ref.joint_name.startswith("right_hand_") or ref.joint_name.startswith("left_hand_"):
        return 4.0, 0.15
    if "wrist_pitch" in ref.joint_name or "wrist_yaw" in ref.joint_name:
        return 25.0, 1.0
    if "shoulder" in ref.joint_name or "elbow" in ref.joint_name or "wrist_roll" in ref.joint_name:
        return 55.0, 2.0
    if ref.joint_name.startswith("waist_"):
        return 80.0, 3.0
    return 120.0, 5.0


def build_targets(
    step: int,
    hold_targets: dict[str, float],
    right_to_center: ArmInterpolator,
    left_to_center: ArmInterpolator,
    right_to_final: ArmInterpolator,
    left_to_final: ArmInterpolator,
    hand_interpolator: ArmInterpolator,
    wait_steps: int,
) -> tuple[dict[str, float], bool, str]:
    targets = dict(hold_targets)
    move_step = step - wait_steps
    if move_step < 0:
        return targets, False, "waiting"

    # Phase 1: both arms move to body center simultaneously
    if not right_to_center.is_done(move_step):
        targets.update(right_to_center.get_target(move_step))
        targets.update(left_to_center.get_target(move_step))
        return targets, False, "phase1_to_center"

    targets.update(right_to_center.get_target(right_to_center.total_steps))
    targets.update(left_to_center.get_target(left_to_center.total_steps))

    # Phase 2: both arms move to final positions simultaneously
    phase2_step = move_step - right_to_center.total_steps
    if not right_to_final.is_done(phase2_step):
        targets.update(right_to_final.get_target(phase2_step))
        targets.update(left_to_final.get_target(phase2_step))
        return targets, False, "phase2_to_final"

    targets.update(right_to_final.get_target(right_to_final.total_steps))
    targets.update(left_to_final.get_target(left_to_final.total_steps))

    # Phase 3: gripper opens
    phase3_step = phase2_step - right_to_final.total_steps
    if not hand_interpolator.is_done(phase3_step):
        targets.update(hand_interpolator.get_target(phase3_step))
        return targets, False, "phase3_gripper_open"

    targets.update(hand_interpolator.get_target(hand_interpolator.total_steps))
    return targets, True, "done"


def check_wrist_orientations(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for side, body, label, target_up in [
        ("right", "right_wrist_yaw_link", "palm DOWN (drop)", -1.0),
        ("left",  "left_wrist_yaw_link",  "basket UP (catch)", +1.0),
    ]:
        bid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        score = float(data.xmat[bid].reshape(3, 3)[:, 1][2])
        ok    = (score * target_up) > 0.7
        status = "OK" if ok else "WARNING"
        print(f"  {side} wrist — {label}: score={score:.3f}  [{status}]")


def print_selected_mapping(refs: dict[str, JointActuator]) -> None:
    print("right arm mapping:")
    for name in RIGHT_ARM_JOINTS:
        ref = refs[name]
        print(
            f"  {name}: actuator={ref.actuator_name}({ref.actuator_id}), "
            f"qpos={ref.qposadr}, qvel={ref.qveladr}"
        )
    print("right hand mapping:")
    for name in RIGHT_HAND_JOINTS:
        ref = refs[name]
        print(
            f"  {name}: actuator={ref.actuator_name}({ref.actuator_id}), "
            f"qpos={ref.qposadr}, qvel={ref.qveladr}"
        )


def run_demo(args: argparse.Namespace) -> None:
    scene_path = Path(args.scene)
    if not scene_path.exists():
        raise FileNotFoundError(
            f"{scene_path} not found. "
            "Get the model: git clone https://github.com/unitreerobotics/unitree_mujoco.git "
            "then pass --scene <path>/unitree_robots/g1/scene_29dof_with_hand.xml"
        )

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = SIMULATE_DT
    if not args.enable_gravity:
        model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)

    all_joint_names = get_actuated_joint_names(model)
    refs = {name: get_joint_actuator(model, name) for name in all_joint_names}
    print_selected_mapping(refs)
    if args.free_base:
        print("floating base: free (robot can fall without a balance controller)")
    else:
        print("floating base: locked for arm-only MuJoCo demo")
    if args.enable_gravity:
        print("gravity: enabled")
    else:
        print("gravity: disabled for stable arm-only demo")
    if args.dynamic:
        print("playback mode: dynamic PD simulation")
    else:
        print("playback mode: kinematic visual replay")

    hand_closed = clamp_targets(RIGHT_HAND_CLOSED_POSE, refs)
    hand_open = clamp_targets(RIGHT_HAND_OPEN_POSE, refs)

    left_basket_pose, left_target_pos, left_hand_pos, left_ik_error_m = (
        solve_left_basket_pose(model, data, refs)
    )
    set_joint_positions(data, refs, left_basket_pose)
    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(args.seed)
    arm_start = sample_right_arm_start(model, data, rng, refs)
    (
        drop_pose,
        left_hand_pos,
        right_target_pos,
        right_elbow_pos,
        ik_error_m,
        right_elbow_error_m,
    ) = solve_right_arm_drop_pose(
        model, data, refs, left_basket_pose
    )

    # load body center poses from JSON, convert keys to _joint suffix for refs
    _pose_json = json.loads(DEFAULT_POSE_JSON.read_text(encoding="utf-8"))
    right_center_pose = {k + "_joint": v for k, v in _pose_json["right_arm_body_center"].items()}
    left_center_pose  = {k + "_joint": v for k, v in _pose_json["left_arm_body_center"].items()}

    # left arm starts at zeros (natural rest) so path is: zeros → center → basket
    left_arm_zeros = {j: 0.0 for j in LEFT_ARM_JOINTS}
    set_joint_positions(data, refs, left_arm_zeros)
    set_joint_positions(data, refs, arm_start)
    set_joint_positions(data, refs, hand_closed)
    mujoco.mj_forward(model, data)
    base_qpos = data.qpos[FREE_BASE_QPOS].copy()

    hold_targets = read_targets(data, refs, all_joint_names)
    center_duration_s = estimate_move_duration_s(arm_start, right_center_pose)
    final_duration_s  = estimate_move_duration_s(right_center_pose, drop_pose)
    steps_per_s = round(1.0 / SIMULATE_DT)

    right_to_center = ArmInterpolator(arm_start,        right_center_pose, center_duration_s, steps_per_s)
    left_to_center  = ArmInterpolator(left_arm_zeros,   left_center_pose,  center_duration_s, steps_per_s)
    right_to_final  = ArmInterpolator(right_center_pose, drop_pose,        final_duration_s,  steps_per_s)
    left_to_final   = ArmInterpolator(left_center_pose,  left_basket_pose, final_duration_s,  steps_per_s)
    hand_interpolator = ArmInterpolator(
        hand_closed, hand_open, GRIPPER_OPEN_DURATION_S, steps_per_s
    )
    wait_steps = round(WAIT_BEFORE_MOVE_S / SIMULATE_DT)
    done_hold_steps = round(HOLD_AFTER_DONE_S / SIMULATE_DT)

    print("right arm random start:")
    for name, value in arm_start.items():
        print(f"  {name}: {value:.4f}")
    print("left basket pose:")
    for name, value in left_basket_pose.items():
        print(f"  {name}: {value:.4f}")
    print(
        "left hand target: "
        f"({left_target_pos[0]:.3f}, {left_target_pos[1]:.3f}, {left_target_pos[2]:.3f})"
    )
    print(
        "left hand position: "
        f"({left_hand_pos[0]:.3f}, {left_hand_pos[1]:.3f}, {left_hand_pos[2]:.3f})"
    )
    print(f"left arm IK error: {left_ik_error_m:.4f} m")
    print(
        "right hand target: "
        f"({right_target_pos[0]:.3f}, {right_target_pos[1]:.3f}, {right_target_pos[2]:.3f})"
    )
    print(f"right arm IK error: {ik_error_m:.4f} m")
    right_elbow_rel = right_elbow_pos - base_qpos[:3]
    print(
        "right elbow position rel pelvis: "
        f"({right_elbow_rel[0]:.3f}, {right_elbow_rel[1]:.3f}, {right_elbow_rel[2]:.3f})"
    )
    print(f"right elbow secondary IK error: {right_elbow_error_m:.4f} m")
    print("right arm drop pose:")
    for name, value in drop_pose.items():
        print(f"  {name}: {value:.4f}")
    set_joint_positions(data, refs, drop_pose)
    set_joint_positions(data, refs, left_basket_pose)
    mujoco.mj_forward(model, data)
    print("wrist orientation check (at final pose):")
    check_wrist_orientations(model, data)
    set_joint_positions(data, refs, arm_start)
    mujoco.mj_forward(model, data)
    print(
        f"phase1 duration: {center_duration_s:.2f}s  phase2 duration: {final_duration_s:.2f}s "
        f"(max joint speed <= {MAX_JOINT_SPEED_RAD_S:.2f} rad/s)"
    )

    if args.no_viewer:
        run_loop(
            model, data, refs, hold_targets,
            right_to_center, left_to_center,
            right_to_final, left_to_final,
            hand_interpolator,
            wait_steps, done_hold_steps,
            base_qpos=base_qpos,
            lock_base=not args.free_base,
            dynamic=args.dynamic,
            viewer=None,
            real_time=False,
        )
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_camera(viewer)
        run_loop(
            model, data, refs, hold_targets,
            right_to_center, left_to_center,
            right_to_final, left_to_final,
            hand_interpolator,
            wait_steps, done_hold_steps=None,
            base_qpos=base_qpos,
            lock_base=not args.free_base,
            dynamic=args.dynamic,
            viewer=viewer,
            real_time=True,
        )


def run_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    refs: dict[str, JointActuator],
    hold_targets: dict[str, float],
    right_to_center: ArmInterpolator,
    left_to_center: ArmInterpolator,
    right_to_final: ArmInterpolator,
    left_to_final: ArmInterpolator,
    hand_interpolator: ArmInterpolator,
    wait_steps: int,
    done_hold_steps: int | None,
    base_qpos: np.ndarray,
    lock_base: bool,
    dynamic: bool,
    viewer,
    real_time: bool,
) -> None:
    step = 0
    done_step = None
    last_viewer_sync = 0.0

    current_phase = "waiting"

    while viewer is None or viewer.is_running():
        step_start = time.perf_counter()
        targets, done, phase = build_targets(
            step, hold_targets,
            right_to_center, left_to_center,
            right_to_final, left_to_final,
            hand_interpolator, wait_steps,
        )

        if phase != current_phase:
            current_phase = phase
            if phase == "phase1_to_center":
                print("→ Phase 1: both arms moving to body center")
            elif phase == "phase2_to_final":
                print("→ Phase 2: both arms moving to final positions")
            elif phase == "phase3_gripper_open":
                print("→ Phase 3: right gripper opening — okra drops into basket")
            elif phase == "done":
                print("→ Done: all phases complete")

        if dynamic:
            apply_pd_control(data, refs, targets)
            mujoco.mj_step(model, data)
            if lock_base:
                lock_floating_base(model, data, base_qpos)
            sim_time = data.time
        else:
            set_joint_positions(data, refs, targets)
            if lock_base:
                data.qpos[FREE_BASE_QPOS] = base_qpos
                data.qvel[FREE_BASE_QVEL] = 0.0
            data.time = step * model.opt.timestep
            mujoco.mj_forward(model, data)
            sim_time = data.time

        if viewer is not None and sim_time - last_viewer_sync >= VIEWER_DT:
            viewer.sync()
            last_viewer_sync = sim_time

        if done and done_step is None:
            done_step = step
            print("drop motion complete; holding final pose")

        step += 1
        if done_step is not None and done_hold_steps is not None:
            if step - done_step >= done_hold_steps:
                break

        if real_time:
            sleep_s = model.opt.timestep - (time.perf_counter() - step_start)
            if sleep_s > 0:
                time.sleep(sleep_s)


def lock_floating_base(
    model: mujoco.MjModel, data: mujoco.MjData, base_qpos: np.ndarray
) -> None:
    data.qpos[FREE_BASE_QPOS] = base_qpos
    data.qvel[FREE_BASE_QVEL] = 0.0
    mujoco.mj_forward(model, data)


def configure_camera(viewer) -> None:
    viewer.cam.lookat[:] = (0.15, 0.0, 0.75)
    viewer.cam.distance = 3.2
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move G1 right arm over the basket and open the right gripper in MuJoCo."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=DEFAULT_SCENE_XML,
        help="path to MuJoCo scene XML (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="run the simulation without opening the MuJoCo viewer",
    )
    parser.add_argument(
        "--free-base",
        action="store_true",
        help="do not lock the floating base; requires a balance controller to stay upright",
    )
    parser.add_argument(
        "--enable-gravity",
        action="store_true",
        help="enable gravity; the default disables it to avoid shaking in the arm-only demo",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="use PD torques and mj_step; default is kinematic visual replay to avoid jitter",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
