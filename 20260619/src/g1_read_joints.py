from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


SDK_PATH = Path("/home/techshare/ILkit/unitree_sdk2_python")
if SDK_PATH.exists() and str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


JOINTS = {
    "left_hip_pitch": 0,
    "left_hip_roll": 1,
    "left_hip_yaw": 2,
    "left_knee": 3,
    "left_ankle_pitch": 4,
    "left_ankle_roll": 5,
    "right_hip_pitch": 6,
    "right_hip_roll": 7,
    "right_hip_yaw": 8,
    "right_knee": 9,
    "right_ankle_pitch": 10,
    "right_ankle_roll": 11,
    "waist_yaw": 12,
    "waist_roll": 13,
    "waist_pitch": 14,
    "left_shoulder_pitch": 15,
    "left_shoulder_roll": 16,
    "left_shoulder_yaw": 17,
    "left_elbow": 18,
    "left_wrist_roll": 19,
    "left_wrist_pitch": 20,
    "left_wrist_yaw": 21,
    "right_shoulder_pitch": 22,
    "right_shoulder_roll": 23,
    "right_shoulder_yaw": 24,
    "right_elbow": 25,
    "right_wrist_roll": 26,
    "right_wrist_pitch": 27,
    "right_wrist_yaw": 28,
}

DEFAULT_JOINT_NAMES = (
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)


class LowStateReader:
    def __init__(self) -> None:
        self.low_state = None
        self.last_update_s = 0.0

    def callback(self, msg: LowState_) -> None:
        self.low_state = msg
        self.last_update_s = time.monotonic()


def format_joint_line(low_state: LowState_, joint_name: str) -> str:
    index = JOINTS[joint_name]
    state = low_state.motor_state[index]
    return (
        f"{index:02d} {joint_name:22s} "
        f"q={state.q: .5f} rad  dq={state.dq: .5f} rad/s  tau={state.tau_est: .5f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read and print G1 joint states only.")
    parser.add_argument("--iface", default="enp8s0", help="DDS network interface")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds; 0 means until Ctrl-C")
    parser.add_argument("--rate", type=float, default=2.0, help="print frequency in Hz")
    parser.add_argument(
        "--all",
        action="store_true",
        help="print all 29 G1 body joints instead of waist and arms only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ChannelFactoryInitialize(args.domain, args.iface)

    reader = LowStateReader()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(reader.callback, 10)

    print(f"Listening to rt/lowstate on iface={args.iface}, domain={args.domain}")
    start_s = time.monotonic()
    while reader.low_state is None:
        if time.monotonic() - start_s > 5.0:
            raise TimeoutError("No LowState received within 5 seconds")
        time.sleep(0.05)

    joint_names = tuple(JOINTS) if args.all else DEFAULT_JOINT_NAMES
    period_s = 1.0 / args.rate
    next_print_s = 0.0

    try:
        while True:
            now_s = time.monotonic()
            if args.duration > 0 and now_s - start_s >= args.duration:
                break
            if now_s >= next_print_s:
                low_state = reader.low_state
                age_ms = (now_s - reader.last_update_s) * 1000.0
                print()
                print(
                    f"mode_machine={getattr(low_state, 'mode_machine', 'n/a')} "
                    f"state_age={age_ms:.1f} ms"
                )
                for joint_name in joint_names:
                    print(format_joint_line(low_state, joint_name))
                next_print_s = now_s + period_s
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped by user")


if __name__ == "__main__":
    main()
