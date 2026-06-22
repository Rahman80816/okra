from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


SDK_PATH = Path(
    os.environ.get(
        "UNITREE_SDK2_PYTHON_PATH", "/home/techshare/ILkit/unitree_sdk2_python"
    )
)
if SDK_PATH.exists() and str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))

from arm_interpolator import ArmInterpolator
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


DEFAULT_TARGET_JSON = (
    Path(__file__).resolve().parent / "data" / "mujoco_right_arm_drop_pose.json"
)


class G1JointIndex:
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    kNotUsedJoint = 29


RIGHT_ARM_JOINTS = {
    "right_shoulder_pitch": G1JointIndex.RightShoulderPitch,
    "right_shoulder_roll": G1JointIndex.RightShoulderRoll,
    "right_shoulder_yaw": G1JointIndex.RightShoulderYaw,
    "right_elbow": G1JointIndex.RightElbow,
    "right_wrist_roll": G1JointIndex.RightWristRoll,
    "right_wrist_pitch": G1JointIndex.RightWristPitch,
    "right_wrist_yaw": G1JointIndex.RightWristYaw,
}

LEFT_ARM_JOINTS = {
    "left_shoulder_pitch": G1JointIndex.LeftShoulderPitch,
    "left_shoulder_roll": G1JointIndex.LeftShoulderRoll,
    "left_shoulder_yaw": G1JointIndex.LeftShoulderYaw,
    "left_elbow": G1JointIndex.LeftElbow,
    "left_wrist_roll": G1JointIndex.LeftWristRoll,
    "left_wrist_pitch": G1JointIndex.LeftWristPitch,
    "left_wrist_yaw": G1JointIndex.LeftWristYaw,
}

DEFAULT_RATE_HZ = 50.0
MIN_MOVE_DURATION_S = 3.0
MAX_ALLOWED_STAGE_DELTA_RAD = 0.15


class LowStateReader:
    def __init__(self) -> None:
        self.low_state = None
        self.last_update_s = 0.0

    def callback(self, msg: LowState_) -> None:
        self.low_state = msg
        self.last_update_s = time.monotonic()


def wait_for_low_state(reader: LowStateReader, timeout_s: float) -> LowState_:
    start_s = time.monotonic()
    while reader.low_state is None:
        if time.monotonic() - start_s > timeout_s:
            raise TimeoutError(f"No LowState received within {timeout_s:.1f} seconds")
        time.sleep(0.02)
    return reader.low_state


def read_right_arm_q(low_state: LowState_) -> dict[str, float]:
    return {
        joint_name: float(low_state.motor_state[index].q)
        for joint_name, index in RIGHT_ARM_JOINTS.items()
    }


def read_left_arm_q(low_state: LowState_) -> dict[str, float]:
    return {
        joint_name: float(low_state.motor_state[index].q)
        for joint_name, index in LEFT_ARM_JOINTS.items()
    }


def _validate_and_extract(section: dict, joints: dict, label: str) -> dict[str, float]:
    missing = set(joints) - set(section)
    extra = set(section) - set(joints)
    if missing or extra:
        raise ValueError(
            f"{label} keys must match joints "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    return {joint_name: float(section[joint_name]) for joint_name in joints}


def load_all_poses(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "right_drop": _validate_and_extract(
            payload["right_arm_drop_pose"], RIGHT_ARM_JOINTS, "right_arm_drop_pose"
        ),
        "left_basket": _validate_and_extract(
            payload["left_basket_pose"], LEFT_ARM_JOINTS, "left_basket_pose"
        ),
        "right_center": _validate_and_extract(
            payload["right_arm_body_center"], RIGHT_ARM_JOINTS, "right_arm_body_center"
        ),
        "left_center": _validate_and_extract(
            payload["left_arm_body_center"], LEFT_ARM_JOINTS, "left_arm_body_center"
        ),
    }


def compute_stage_goal(
    start_q: dict[str, float],
    target_q: dict[str, float],
    fraction: float,
    max_joint_delta: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    stage_goal = {}
    unclamped_delta = {}
    applied_delta = {}
    for joint_name in RIGHT_ARM_JOINTS:
        raw_delta = (target_q[joint_name] - start_q[joint_name]) * fraction
        clipped_delta = max(-max_joint_delta, min(max_joint_delta, raw_delta))
        stage_goal[joint_name] = start_q[joint_name] + clipped_delta
        unclamped_delta[joint_name] = raw_delta
        applied_delta[joint_name] = clipped_delta
    return stage_goal, unclamped_delta, applied_delta


def fill_right_arm_cmd(
    low_cmd,
    target_q: dict[str, float],
    kp: float,
    kd: float,
    mode_machine: int,
) -> None:
    low_cmd.mode_pr = 0
    low_cmd.mode_machine = mode_machine
    low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0
    for joint_name, index in RIGHT_ARM_JOINTS.items():
        motor_cmd = low_cmd.motor_cmd[index]
        motor_cmd.mode = 1
        motor_cmd.tau = 0.0
        motor_cmd.q = target_q[joint_name]
        motor_cmd.dq = 0.0
        motor_cmd.kp = kp
        motor_cmd.kd = kd


def release_arm_sdk(
    publisher: ChannelPublisher,
    crc: CRC,
    mode_machine: int,
    repeat: int = 20,
) -> None:
    low_cmd = unitree_hg_msg_dds__LowCmd_()
    low_cmd.mode_pr = 0
    low_cmd.mode_machine = mode_machine
    low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 0.0
    for _ in range(repeat):
        low_cmd.crc = crc.Crc(low_cmd)
        publisher.Write(low_cmd)
        time.sleep(0.02)


def publish_target(
    publisher: ChannelPublisher,
    crc: CRC,
    low_cmd,
    target_q: dict[str, float],
    kp: float,
    kd: float,
    mode_machine: int,
) -> None:
    fill_right_arm_cmd(low_cmd, target_q, kp=kp, kd=kd, mode_machine=mode_machine)
    low_cmd.crc = crc.Crc(low_cmd)
    publisher.Write(low_cmd)


def max_abs_error(actual_q: dict[str, float], target_q: dict[str, float]) -> tuple[str, float]:
    joint_name = max(
        RIGHT_ARM_JOINTS,
        key=lambda name: abs(actual_q[name] - target_q[name]),
    )
    return joint_name, abs(actual_q[joint_name] - target_q[joint_name])


def print_tracking_line(
    label: str,
    elapsed_s: float,
    reader: LowStateReader,
    target_q: dict[str, float],
) -> None:
    low_state = reader.low_state
    if low_state is None:
        print(f"{label} t={elapsed_s:5.2f}s no lowstate yet")
        return
    actual_q = read_right_arm_q(low_state)
    joint_name, error = max_abs_error(actual_q, target_q)
    age_ms = (time.monotonic() - reader.last_update_s) * 1000.0
    print(
        f"{label} t={elapsed_s:5.2f}s "
        f"max_err={error:.4f} rad ({joint_name}) "
        f"state_age={age_ms:.1f} ms"
    )


def run_interpolator(
    publisher: ChannelPublisher,
    crc: CRC,
    low_cmd,
    reader: LowStateReader,
    start_q: dict[str, float],
    goal_q: dict[str, float],
    duration_s: float,
    rate_hz: float,
    log_rate_hz: float,
    kp: float,
    kd: float,
    mode_machine: int,
    label: str,
) -> None:
    interpolator = ArmInterpolator(start_q, goal_q, duration_s, round(rate_hz))
    period_s = 1.0 / rate_hz
    log_period_s = 1.0 / log_rate_hz if log_rate_hz > 0 else None
    next_log_s = 0.0
    start_s = time.monotonic()
    print(f"{label}: {duration_s:.2f}s, {interpolator.total_steps} steps")
    for step in range(interpolator.total_steps + 1):
        step_start_s = time.monotonic()
        target_q = interpolator.get_target(step)
        publish_target(
            publisher,
            crc,
            low_cmd,
            target_q,
            kp=kp,
            kd=kd,
            mode_machine=mode_machine,
        )
        elapsed_s = step_start_s - start_s
        if log_period_s is not None and elapsed_s >= next_log_s:
            print_tracking_line(label, elapsed_s, reader, target_q)
            next_log_s += log_period_s
        sleep_s = period_s - (time.monotonic() - step_start_s)
        if sleep_s > 0:
            time.sleep(sleep_s)


def confirm_or_exit(
    args: argparse.Namespace,
    start_q: dict[str, float],
    target_q: dict[str, float],
    stage_goal: dict[str, float],
    unclamped_delta: dict[str, float],
    applied_delta: dict[str, float],
) -> None:
    print(f"Target JSON: {args.target_json}")
    print(f"fraction={args.fraction:.3f}, max_joint_delta={args.max_joint_delta:.3f} rad")
    print("Planned staged move:")
    for joint_name in RIGHT_ARM_JOINTS:
        clamp_mark = " clipped" if abs(unclamped_delta[joint_name] - applied_delta[joint_name]) > 1e-9 else ""
        print(
            f"  {joint_name:22s} "
            f"start={start_q[joint_name]: .5f} "
            f"mujoco_target={target_q[joint_name]: .5f} "
            f"stage_goal={stage_goal[joint_name]: .5f} "
            f"delta={applied_delta[joint_name]: .5f}{clamp_mark}"
        )
    if args.keep_stage:
        print("The arm will stay at the staged pose and arm_sdk will be released.")
    else:
        print("The arm will return to the captured start posture after the staged pose.")
    if args.yes:
        return
    answer = input("Type STAGE to continue: ").strip()
    if answer != "STAGE":
        raise SystemExit("Canceled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move the real G1 right arm a small staged fraction toward a target pose."
    )
    parser.add_argument("--iface", default="enp8s0", help="DDS network interface")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id")
    parser.add_argument("--target-json", type=Path, default=DEFAULT_TARGET_JSON)
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.15,
        help="fraction of current-to-target difference to apply this run",
    )
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.08,
        help="maximum absolute per-joint delta applied in this run",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="move duration in seconds")
    parser.add_argument("--hold", type=float, default=1.0, help="hold time at staged pose")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, help="command frequency in Hz")
    parser.add_argument("--log-rate", type=float, default=2.0, help="tracking print frequency in Hz")
    parser.add_argument("--kp", type=float, default=15.0, help="right arm kp")
    parser.add_argument("--kd", type=float, default=1.0, help="right arm kd")
    parser.add_argument("--keep-stage", action="store_true", help="do not return to the captured start posture")
    parser.add_argument("--yes", action="store_true", help="skip interactive confirmation")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.target_json.exists():
        raise FileNotFoundError(args.target_json)
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")
    if not 0.0 < args.max_joint_delta <= MAX_ALLOWED_STAGE_DELTA_RAD:
        raise ValueError(
            f"--max-joint-delta must be in (0, {MAX_ALLOWED_STAGE_DELTA_RAD:.2f}]"
        )
    if args.duration < MIN_MOVE_DURATION_S:
        raise ValueError(f"--duration must be >= {MIN_MOVE_DURATION_S:.1f}s")
    if args.hold < 0:
        raise ValueError("--hold must be non-negative")
    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    if args.log_rate < 0:
        raise ValueError("--log-rate must be non-negative")
    if args.kp < 0 or args.kd < 0:
        raise ValueError("--kp and --kd must be non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    target_q = load_target_pose(args.target_json)

    ChannelFactoryInitialize(args.domain, args.iface)

    reader = LowStateReader()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(reader.callback, 10)

    publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    publisher.Init()
    crc = CRC()

    low_state = wait_for_low_state(reader, timeout_s=5.0)
    mode_machine = int(getattr(low_state, "mode_machine", 0))
    start_q = read_right_arm_q(low_state)
    stage_goal, unclamped_delta, applied_delta = compute_stage_goal(
        start_q,
        target_q,
        fraction=args.fraction,
        max_joint_delta=args.max_joint_delta,
    )

    confirm_or_exit(args, start_q, target_q, stage_goal, unclamped_delta, applied_delta)

    low_cmd = unitree_hg_msg_dds__LowCmd_()
    try:
        run_interpolator(
            publisher,
            crc,
            low_cmd,
            reader,
            start_q,
            stage_goal,
            duration_s=args.duration,
            rate_hz=args.rate,
            log_rate_hz=args.log_rate,
            kp=args.kp,
            kd=args.kd,
            mode_machine=mode_machine,
            label="move toward target stage",
        )

        hold_until_s = time.monotonic() + args.hold
        while time.monotonic() < hold_until_s:
            publish_target(
                publisher,
                crc,
                low_cmd,
                stage_goal,
                kp=args.kp,
                kd=args.kd,
                mode_machine=mode_machine,
            )
            time.sleep(1.0 / args.rate)
        if args.hold > 0:
            print_tracking_line("hold stage", args.hold, reader, stage_goal)

        if not args.keep_stage:
            current_q = read_right_arm_q(wait_for_low_state(reader, timeout_s=1.0))
            run_interpolator(
                publisher,
                crc,
                low_cmd,
                reader,
                current_q,
                start_q,
                duration_s=args.duration,
                rate_hz=args.rate,
                log_rate_hz=args.log_rate,
                kp=args.kp,
                kd=args.kd,
                mode_machine=mode_machine,
                label="return to start",
            )
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        print("Releasing arm_sdk")
        release_arm_sdk(publisher, crc, mode_machine=mode_machine)


if __name__ == "__main__":
    main()
