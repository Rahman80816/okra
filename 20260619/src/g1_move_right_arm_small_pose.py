from __future__ import annotations

import argparse
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


class G1JointIndex:
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

DEFAULT_DELTAS = {
    "right_shoulder_pitch": -0.02,
    "right_shoulder_roll": 0.02,
    "right_shoulder_yaw": 0.02,
    "right_elbow": 0.04,
    "right_wrist_roll": 0.03,
    "right_wrist_pitch": -0.03,
    "right_wrist_yaw": 0.03,
}

DEFAULT_RATE_HZ = 50.0
MAX_ABS_DELTA_RAD = 0.08
MIN_MOVE_DURATION_S = 3.0


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


def scaled_deltas(scale: float) -> dict[str, float]:
    return {joint_name: delta * scale for joint_name, delta in DEFAULT_DELTAS.items()}


def fill_right_arm_cmd(low_cmd, target_q: dict[str, float], kp: float, kd: float) -> None:
    low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0
    for joint_name, index in RIGHT_ARM_JOINTS.items():
        motor_cmd = low_cmd.motor_cmd[index]
        motor_cmd.tau = 0.0
        motor_cmd.q = target_q[joint_name]
        motor_cmd.dq = 0.0
        motor_cmd.kp = kp
        motor_cmd.kd = kd


def release_arm_sdk(publisher: ChannelPublisher, crc: CRC, repeat: int = 20) -> None:
    low_cmd = unitree_hg_msg_dds__LowCmd_()
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
) -> None:
    fill_right_arm_cmd(low_cmd, target_q, kp=kp, kd=kd)
    low_cmd.crc = crc.Crc(low_cmd)
    publisher.Write(low_cmd)


def run_interpolator(
    publisher: ChannelPublisher,
    crc: CRC,
    low_cmd,
    start_q: dict[str, float],
    goal_q: dict[str, float],
    duration_s: float,
    rate_hz: float,
    kp: float,
    kd: float,
    label: str,
) -> None:
    interpolator = ArmInterpolator(start_q, goal_q, duration_s, round(rate_hz))
    period_s = 1.0 / rate_hz
    print(f"{label}: {duration_s:.2f}s, {interpolator.total_steps} steps")
    for step in range(interpolator.total_steps + 1):
        step_start_s = time.monotonic()
        publish_target(
            publisher,
            crc,
            low_cmd,
            interpolator.get_target(step),
            kp=kp,
            kd=kd,
        )
        sleep_s = period_s - (time.monotonic() - step_start_s)
        if sleep_s > 0:
            time.sleep(sleep_s)


def confirm_or_exit(
    args: argparse.Namespace,
    start_q: dict[str, float],
    goal_q: dict[str, float],
    deltas: dict[str, float],
) -> None:
    if args.yes:
        return
    print("This script will publish to rt/arm_sdk and move all 7 RIGHT arm joints slightly.")
    print("Planned deltas:")
    for joint_name in RIGHT_ARM_JOINTS:
        print(
            f"  {joint_name:22s} "
            f"start={start_q[joint_name]: .5f} "
            f"goal={goal_q[joint_name]: .5f} "
            f"delta={deltas[joint_name]: .5f}"
        )
    print("The arm will return to the captured start posture unless --no-return is set.")
    answer = input("Type MOVE7 to continue: ").strip()
    if answer != "MOVE7":
        raise SystemExit("Canceled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move all 7 G1 right arm joints by tiny deltas through rt/arm_sdk."
    )
    parser.add_argument("--iface", default="enp8s0", help="DDS network interface")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id")
    parser.add_argument("--scale", type=float, default=1.0, help="scale default small deltas")
    parser.add_argument("--duration", type=float, default=4.0, help="move duration in seconds")
    parser.add_argument("--hold", type=float, default=1.0, help="hold time at the goal in seconds")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, help="command frequency in Hz")
    parser.add_argument("--kp", type=float, default=15.0, help="right arm kp")
    parser.add_argument("--kd", type=float, default=1.0, help="right arm kd")
    parser.add_argument("--no-return", action="store_true", help="do not return to the captured start posture")
    parser.add_argument("--yes", action="store_true", help="skip interactive confirmation")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    deltas = scaled_deltas(args.scale)
    max_delta = max(abs(delta) for delta in deltas.values())
    if max_delta > MAX_ABS_DELTA_RAD:
        raise ValueError(
            f"scaled deltas must be within +/-{MAX_ABS_DELTA_RAD:.2f} rad "
            f"(got max {max_delta:.3f})"
        )
    if args.duration < MIN_MOVE_DURATION_S:
        raise ValueError(f"--duration must be >= {MIN_MOVE_DURATION_S:.1f}s")
    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    if args.kp < 0 or args.kd < 0:
        raise ValueError("--kp and --kd must be non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)

    ChannelFactoryInitialize(args.domain, args.iface)

    reader = LowStateReader()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(reader.callback, 10)

    publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    publisher.Init()
    crc = CRC()

    low_state = wait_for_low_state(reader, timeout_s=5.0)
    start_q = read_right_arm_q(low_state)
    deltas = scaled_deltas(args.scale)
    goal_q = {
        joint_name: start_q[joint_name] + deltas[joint_name]
        for joint_name in RIGHT_ARM_JOINTS
    }

    confirm_or_exit(args, start_q, goal_q, deltas)

    low_cmd = unitree_hg_msg_dds__LowCmd_()
    try:
        run_interpolator(
            publisher,
            crc,
            low_cmd,
            start_q,
            goal_q,
            duration_s=args.duration,
            rate_hz=args.rate,
            kp=args.kp,
            kd=args.kd,
            label="move to small 7-joint offset",
        )

        hold_until_s = time.monotonic() + args.hold
        while time.monotonic() < hold_until_s:
            publish_target(publisher, crc, low_cmd, goal_q, kp=args.kp, kd=args.kd)
            time.sleep(1.0 / args.rate)

        if not args.no_return:
            run_interpolator(
                publisher,
                crc,
                low_cmd,
                goal_q,
                start_q,
                duration_s=args.duration,
                rate_hz=args.rate,
                kp=args.kp,
                kd=args.kd,
                label="return to start",
            )
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        print("Releasing arm_sdk")
        release_arm_sdk(publisher, crc)


if __name__ == "__main__":
    main()
