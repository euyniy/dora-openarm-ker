# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""dora-rs node for leader OpenArm KER."""

import argparse
import json
import time
import dora
import pyarrow as pa
import numpy as np
from openarm_ker.ker_stream import KERStream, CMD_STANDBY, CMD_STREAM

QPOS_TYPE = pa.struct([("qpos", pa.list_(pa.float32()))])


def build_qpos_output(qpos: np.ndarray) -> pa.Array:
    """Wrap a qpos array as a length-1 StructArray: [{"qpos": [...]}]."""
    return pa.array([{"qpos": qpos}], type=QPOS_TYPE)


# ==============================================================================
# Filters & Math Utilities
# ==============================================================================
def map_range(x, in_min, in_max, out_min, out_max):
    """Map a value from one range to another, with clipping."""
    if in_max == in_min:
        return out_min
    x = max(min(x, in_max), in_min) if in_min < in_max else max(min(x, in_min), in_max)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class OnlineHampelFilter:
    """Real-time Hampel filter to eliminate hardware spikes with zero phase delay."""

    def __init__(self, window_size=5, n_sigmas=3.0, min_threshold=5.0):
        """Initialize the Hampel filter parameters and historical buffer."""
        self.window_size = window_size
        self.n_sigmas = n_sigmas
        self.min_threshold = min_threshold
        self.scale_factor = 1.4826
        self.history = None

    def process(self, current_values: list[float]) -> list[float]:
        """Process streaming multi-channel data to detect and suppress sudden spikes."""
        curr = np.array(current_values, dtype=np.float64)

        if self.history is None:
            self.history = np.tile(curr, (self.window_size, 1))
            return current_values

        medians = np.median(self.history, axis=0)
        mads = np.median(np.abs(self.history - medians), axis=0)

        thresholds = np.maximum(
            self.n_sigmas * self.scale_factor * mads, self.min_threshold
        )

        is_outlier = np.abs(curr - medians) > thresholds
        if np.any(is_outlier):
            for idx in np.where(is_outlier)[0]:
                print(
                    f"[KER Filter] Spike suppressed at J{idx + 1}! Raw: {curr[idx]:.2f} -> Fixed: {medians[idx]:.2f}"
                )

        filtered_values = np.where(is_outlier, medians, curr)

        self.history[:-1] = self.history[1:]
        self.history[-1] = filtered_values

        return filtered_values.tolist()


class OneEuroFilter:
    """Per-channel 1 Euro Filter: smooths jitter at rest, follows fast motion.

    The cutoff frequency rises with the filtered speed:
    cutoff = min_cutoff + beta * |velocity|. Units follow the input
    (here rad and rad/s).
    """

    def __init__(self, min_cutoff=5.0, beta=5.0, d_cutoff=5.0):
        """Initialize the filter parameters and empty state."""
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def process(self, t: float, values: np.ndarray) -> np.ndarray:
        """Filter one multi-channel sample taken at monotonic time t (seconds)."""
        x = np.asarray(values, dtype=np.float64)

        if self.t_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0.0:
            return self.x_prev.copy()

        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._alpha(dt, cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat.copy()


# ==============================================================================
# Core Processor
# ==============================================================================
class KerPoseProcessor:
    """Processes raw angles and applies filters without modifying the physical scale."""

    def __init__(
        self,
        use_hampel: bool = False,
        one_euro: OneEuroFilter | None = None,
    ):
        """Initialize the pose processor with optional Hampel and 1 Euro filters."""
        self.hampel = (
            OnlineHampelFilter(window_size=5, n_sigmas=3.0, min_threshold=5.0)
            if use_hampel
            else None
        )
        self.one_euro = one_euro

    def process(
        self, raw_angles: list[float], t: float | None = None
    ) -> tuple[list[float], list[float]]:
        """Filter raw encoder degrees and transform them into left/right radian lists."""
        filtered = self.hampel.process(raw_angles) if self.hampel else raw_angles

        grip_r_deg = map_range(filtered[7], 0.0, -60.0, -60.0, 10.0)
        grip_l_deg = map_range(filtered[15], 0.0, 60.0, 60.0, -10.0)

        pos_right = np.deg2rad(filtered[:7]).tolist()
        pos_right.append(np.deg2rad(grip_r_deg))

        pos_left = np.deg2rad(filtered[8:15]).tolist()
        pos_left.append(np.deg2rad(grip_l_deg))

        if self.one_euro is not None:
            t = time.perf_counter() if t is None else t
            # Arm joints only: grippers stay raw for grasp timing.
            smoothed = self.one_euro.process(t, pos_right[:7] + pos_left[:7]).tolist()
            pos_right = smoothed[:7] + pos_right[7:]
            pos_left = smoothed[7:] + pos_left[7:]

        return pos_right, pos_left


# ==============================================================================
# Main Node App
# ==============================================================================
def main():
    """Act OpenArm KER as a leader of OpenArm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hampel", action="store_true", help="Enable Hampel filter")
    parser.add_argument(
        "--one-euro", action="store_true", help="Enable 1 Euro filter on joints"
    )
    parser.add_argument(
        "--min-cutoff",
        type=float,
        default=5.0,
        help="1 Euro cutoff in Hz at rest; higher = less lag, more jitter "
        "(default: 5.0).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=5.0,
        help="1 Euro cutoff increase in Hz per rad/s; higher = less lag "
        "when moving (default: 5.0).",
    )
    parser.add_argument(
        "--d-cutoff",
        type=float,
        default=5.0,
        help="1 Euro cutoff in Hz for the velocity estimate (default: 5.0).",
    )
    args = parser.parse_args()

    print("Connecting to KER device...")
    stream = KERStream()
    stream.connect()
    stream.send_command(CMD_STREAM)

    print("\n=== Verified Device Metadata ===")
    print(f" Hardware : {stream.metadata.get('hw')}")
    print(f" Firmware : {stream.metadata.get('fw')}")
    print(f" Updated  : {stream.metadata.get('updated')}")
    print("================================\n")

    print("Initializing dora-rs node...")
    node = dora.Node()
    one_euro = (
        OneEuroFilter(
            min_cutoff=args.min_cutoff, beta=args.beta, d_cutoff=args.d_cutoff
        )
        if args.one_euro
        else None
    )
    processor = KerPoseProcessor(use_hampel=args.hampel, one_euro=one_euro)

    print("KER Leader Node Running successfully.\n")
    ### send metadata
    node.send_output(
        "metadata",
        pa.array([json.dumps(stream.metadata)]),
        {"timestamp": time.time_ns()},
    )
    try:
        for event in node:
            if event["type"] == "ERROR":
                print(f"Dora error occurred: {event['error']}")
                break

            elif event["type"] == "STOP":
                print("Received STOP event from dora. Shutting down...")
                break

            if event["type"] != "INPUT":
                continue

            data = stream.recv()
            if data is None:
                continue

            pos_right, pos_left = processor.process(data["angles"])

            ts = {"timestamp": time.time_ns()}

            node.send_output(
                "follower_position_right",
                build_qpos_output(np.array(pos_right, dtype=np.float32)),
                ts,
            )
            node.send_output(
                "follower_position_left",
                build_qpos_output(np.array(pos_left, dtype=np.float32)),
                ts,
            )

            # enc_val = data["encoder_value"]
            # enc_btn = data["encoder_button"]
            #
            # node.send_output("encoder_value", pa.array([enc_val], type=pa.int32()), ts)
            # node.send_output(
            #     "encoder_button", pa.array([int(enc_btn)], type=pa.int32()), ts
            # )
            #
    except KeyboardInterrupt:
        pass

    finally:
        try:
            print("\nShutting down: Sending STANDBY command...")
            stream.send_command(CMD_STANDBY)
            time.sleep(0.1)
        except Exception:
            pass
        stream.close()
        print("KER Leader Node Disconnected safely.")


if __name__ == "__main__":
    main()
