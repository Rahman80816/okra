from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


SDK_PATH = Path("/home/techshare/ILkit/unitree_sdk2_python")
if SDK_PATH.exists() and str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))

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


def fill_right_arm_hold_cmd(
    low_cmd,
    target_q: dict[str, float],
    kp: float,
    kd: float,
) -> None:
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


def confirm_or_exit(args: argparse.Namespace) -> None:
    if args.yes:
        return
    print("This script publishes to rt/arm_sdk and holds the current RIGHT arm posture.")
    print("It should not move the arm, but it will enable arm_sdk while running.")
    answer = input("Type HOLD to continue: ").strip()
    if answer != "HOLD":
        raise SystemExit("Canceled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hold the current G1 right arm posture through rt/arm_sdk."
    )
    parser.add_argument("--iface", default="enp8s0", help="DDS network interface")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id")
    parser.add_argument("--duration", type=float, default=10.0, help="hold seconds; 0 means until Ctrl-C")
    parser.add_argument("--rate", type=float, default=50.0, help="command frequency in Hz")
    parser.add_argument("--kp", type=float, default=20.0, help="right arm hold kp")
    parser.add_argument("--kd", type=float, default=1.0, help="right arm hold kd")
    parser.add_argument("--yes", action="store_true", help="skip interactive confirmation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    confirm_or_exit(args)

    ChannelFactoryInitialize(args.domain, args.iface)

    reader = LowStateReader()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(reader.callback, 10)

    publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    publisher.Init()
    crc = CRC()

    low_state = wait_for_low_state(reader, timeout_s=5.0)
    target_q = read_right_arm_q(low_state)
    print("Captured current right arm posture:")
    for joint_name, q in target_q.items():
        print(f"  {joint_name:22s} {q: .5f} rad")

    low_cmd = unitree_hg_msg_dds__LowCmd_()
    period_s = 1.0 / args.rate
    start_s = time.monotonic()
    next_report_s = start_s

    try:
        while True:
            now_s = time.monotonic()
            if args.duration > 0 and now_s - start_s >= args.duration:
                break

            fill_right_arm_hold_cmd(low_cmd, target_q, kp=args.kp, kd=args.kd)
            low_cmd.crc = crc.Crc(low_cmd)
            publisher.Write(low_cmd)

            if now_s >= next_report_s:
                age_ms = (now_s - reader.last_update_s) * 1000.0
                print(f"holding right arm... t={now_s - start_s:.1f}s state_age={age_ms:.1f}ms")
                next_report_s = now_s + 1.0

            sleep_s = period_s - (time.monotonic() - now_s)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        print("Releasing arm_sdk")
        release_arm_sdk(publisher, crc)


if __name__ == "__main__":
    main()
