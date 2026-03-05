#!/usr/bin/env python3
import argparse
import io
import os
import time

import numpy as np
from ruamel.yaml import YAML

from flightgym import QuadrotorVisEnv_v1
from tonedio_baselines.envs import vis_vec_env_wrapper as wrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manual control for vectorized visual Flightmare env (no RL policy)."
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control_step", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means infinite")
    parser.add_argument("--show_image", type=int, default=1, help="1: cv2 window")
    parser.add_argument("--display_scale", type=int, default=4, help="Display magnification for the 84x84 image")
    parser.add_argument("--step_dt", type=float, default=0.01, help="Target seconds per env.step (0 disables pacing)")
    parser.add_argument("--fps", type=float, default=0.0, help="If >0, overrides step_dt with 1/fps")
    return parser.parse_args()


def build_env(num_envs: int, num_threads: int):
    yaml = YAML()
    cfg_path = os.path.join(os.environ["FLIGHTMARE_PATH"], "flightlib/configs/vec_env.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f)

    cfg["env"]["num_envs"] = max(1, int(num_envs))
    cfg["env"]["num_threads"] = max(1, int(num_threads))
    cfg["env"]["render"] = "yes"

    stream = io.StringIO()
    yaml.dump(cfg, stream)
    cfg_yaml_str = stream.getvalue()

    env = wrapper.VisFlightEnvVec(QuadrotorVisEnv_v1(cfg_yaml_str, False), use_obs_norm=False)
    return env


def clip_action(action):
    return np.clip(action, -1.0, 1.0)


def print_help():
    print("\nManual control keys")
    print("  w/s : throttle up/down (all motors)")
    print("  i/k : pitch forward/back")
    print("  j/l : roll left/right")
    print("  u/o : yaw left/right")
    print("  x   : zero action")
    print("  r   : reset env")
    print("  q   : quit")


def update_action_from_key(action, key, delta):
    # motor order assumed [m0, m1, m2, m3]
    if key == ord("w"):
        action += delta
    elif key == ord("s"):
        action -= delta
    elif key == ord("i"):
        action[0] -= delta
        action[1] -= delta
        action[2] += delta
        action[3] += delta
    elif key == ord("k"):
        action[0] += delta
        action[1] += delta
        action[2] -= delta
        action[3] -= delta
    elif key == ord("j"):
        action[0] -= delta
        action[3] -= delta
        action[1] += delta
        action[2] += delta
    elif key == ord("l"):
        action[0] += delta
        action[3] += delta
        action[1] -= delta
        action[2] -= delta
    elif key == ord("u"):
        action[0] += delta
        action[2] += delta
        action[1] -= delta
        action[3] -= delta
    elif key == ord("o"):
        action[0] -= delta
        action[2] -= delta
        action[1] += delta
        action[3] += delta
    elif key == ord("x"):
        action[:] = 0.0
    return clip_action(action)


def main():
    args = parse_args()
    cv2 = None
    if args.show_image:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "show_image=1 requires OpenCV. Install with: pip install opencv-python"
            ) from e

    env = build_env(args.num_envs, args.num_threads)
    env.seed(args.seed)

    if not env.connectUnity():
        raise RuntimeError("Failed to connect Unity. Launch Unity first.")

    # Useful for manual long runs if underlying env supports it.
    if hasattr(env.wrapper, "setTruncationEnabled"):
        env.wrapper.setTruncationEnabled(False)

    obs = env.reset()
    if obs.ndim != 4:
        raise RuntimeError(f"Expected image observation [N,H,W,C], got {obs.shape}")

    act_dim = env.action_space.shape[0]
    action_single = np.zeros((act_dim,), dtype=np.float32)
    action_batch = np.zeros((env.num_envs, act_dim), dtype=np.float32)

    print_help()
    print(f"obs shape: {obs.shape}, action shape per env: {action_single.shape}")

    step_count = 0
    target_dt = float(args.step_dt)
    if float(args.fps) > 0.0:
        target_dt = 1.0 / float(args.fps)
    if target_dt < 0.0:
        target_dt = 0.0
    print(f"loop pacing: step_dt={target_dt:.4f}s ({(1.0/target_dt):.1f} Hz)" if target_dt > 0 else "loop pacing: disabled (max speed)")

    try:
        while True:
            t0 = time.perf_counter()
            action_batch[:] = action_single[None, :]

            obs, reward, done, info = env.step(action_batch)
            step_count += 1

            if args.show_image:
                frame = obs[0]
                scale = max(1, int(args.display_scale))
                display = cv2.resize(
                    frame,
                    (frame.shape[1] * scale, frame.shape[0] * scale),
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imshow("flightmare_obs_env0", display)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1

            if key == ord("q"):
                break
            if key == ord("r"):
                obs = env.reset()
                print("[Reset] environment reset")
                continue
            if key != -1:
                action_single = update_action_from_key(
                    action_single, key, float(args.control_step)
                )
                print(
                    f"[Step {step_count}] action={np.round(action_single, 3)} "
                    f"reward0={float(reward[0]): .3f} done0={bool(done[0])}"
                )

            if bool(done[0]):
                print(f"[Done] step={step_count}, auto-reset env")
                obs = env.reset()

            if args.max_steps > 0 and step_count >= args.max_steps:
                break

            if target_dt > 0.0:
                elapsed = time.perf_counter() - t0
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
        try:
            env.disconnectUnity()
        finally:
            env.close()


if __name__ == "__main__":
    main()
