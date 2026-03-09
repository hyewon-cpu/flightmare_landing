#!/usr/bin/env python3
from ruamel.yaml import YAML

#
import os
import io
import math
import argparse
import copy
import numpy as np
import torch
import cv2
import datetime
import sys

# Ensure `tonedio_baselines` is importable when running from `examples/`.
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
#
# from stable_baselines import logger

#
# from rpg_baselines.common.policies import MlpPolicy
# from rpg_baselines.ppo.ppo2 import PPO2
# from rpg_baselines.ppo.ppo2_test import test_model
import tonedio_baselines.envs.dot_vec_env_wrapper as wrapper
from tonedio_baselines.envs.dot_vec_env_wrapper import (
    ObsNormUpdateCallback,
    CheckpointCallbackWithRMS
)
import tonedio_baselines.common.util as U
#
from flightgym import QuadrotorDotEnv_v1

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback, BaseCallback

import wandb
from wandb.integration.sb3 import WandbCallback

"""
python run_drone_control.py --train 1 --use_obs_norm 1 --render 0 --wandb_run_name "test_run_1"


"""


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_flightmare_path():
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    )
    env_root = os.environ.get("FLIGHTMARE_PATH", "")
    env_cfg = os.path.join(env_root, "flightlib", "configs", "quadrotor_env.yaml")
    if not env_root or not os.path.isfile(env_cfg):
        os.environ["FLIGHTMARE_PATH"] = root_dir
        print(f"[Config] Set FLIGHTMARE_PATH -> {root_dir}")


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('-w', '--weight', type=str, default=None,
                        help='trained weight path name')
    
    # eval freq, model_save_freq 모두 timestep 기준
    parser.add_argument('--total_timesteps', type=int, default=25_000_000,
                   help="Total training timesteps")
    parser.add_argument('--eval_freq', type=int, default=10_000,
                   help="Eval frequency (timesteps per env)")
    parser.add_argument('--n_eval_episodes', type=int, default=5,
                   help="Number of eval episodes")
    parser.add_argument('--checkpoint_freq', type=int, default=5_000_000, 
                   help="Checkpoint save frequency in total timesteps. Default is 50,000.")

    # wandb
    parser.add_argument('--wandb', type=int, default=1, help="Enable wandb logging")
    parser.add_argument('--wandb_project', type=str, default='flightmare_ppo', help="wandb project name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="wandb run name")
    parser.add_argument('--use_obs_norm', type=int, default=1, help="Use observation normalization (1=True, 0=False)")
    parser.add_argument('--rms_path', type=str, default=None, 
                        help="Path to normalization statistics (.npz file) for testing. "
                             "If None, will try to find RMS file from checkpoint directory.")
    parser.add_argument('--viz_scene', type=int, default=0,
                        help="Visualize observation image + projection in one combined window during train/test")
    parser.add_argument('--proj_img_width', type=int, default=256,
                        help="Projection visualization image width in pixels")
    parser.add_argument('--proj_img_height', type=int, default=256,
                        help="Projection visualization image height in pixels")
    parser.add_argument('--save_obs_image', type=int, default=0,
                        help="Save observation RGB images during training/testing (1=True, 0=False)")
    parser.add_argument('--obs_image_save_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                        help="Directory to save observation images (default: tonediorl/examples/images)")
    parser.add_argument('--obs_image_save_every', type=int, default=10,
                        help="Save one image every N callback/test steps")
    parser.add_argument('--init_pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'), default=None,
                        help="Override quadrotor initial position in meters, e.g. --init_pos 0 0 10")
    return parser


def apply_init_pos_override(init_pos):
    if init_pos is None:
        return
    root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
    quad_cfg_path = os.path.join(root_dir, "flightlib", "configs", "quadrotor_env.yaml")
    yaml = YAML()
    with open(quad_cfg_path, "r") as f:
        quad_cfg = yaml.load(f)
    quad_cfg["quadrotor_env"]["init_pos"] = [float(init_pos[0]), float(init_pos[1]), float(init_pos[2])]
    with open(quad_cfg_path, "w") as f:
        yaml.dump(quad_cfg, f)
    print(f"[Config] Overrode quadrotor_env.init_pos -> {quad_cfg['quadrotor_env']['init_pos']}")

def build_env(cfg_yaml_str, use_obs_norm=True):
    env = wrapper.DotFlightEnvVec(
        QuadrotorDotEnv_v1(cfg_yaml_str, False), use_obs_norm=use_obs_norm
    )
    env = VecMonitor(env)  # episode stats logging
    # VecMonitor는 episode가 끝날 때마다 정보 업데이트. 
    return env


def _parse_tag_and_image(raw_obs_flat):
    raw_obs_flat = np.asarray(raw_obs_flat, dtype=np.float32).reshape(-1)
    img_size = 84 * 84 * 3
    if raw_obs_flat.shape[0] < (11 + img_size):
        return None, None
    tag_dim = raw_obs_flat.shape[0] - img_size
    tag = raw_obs_flat[:tag_dim]
    img_flat = np.clip(raw_obs_flat[tag_dim:tag_dim + img_size], 0.0, 255.0).astype(np.uint8)
    img = img_flat.reshape(84, 84, 3)
    return tag, img


def _split_tags(tag_flat):
    tag_flat = np.asarray(tag_flat, dtype=np.float32).reshape(-1)
    tag_feat = 11
    if tag_flat.shape[0] < tag_feat:
        return []
    n_tags = tag_flat.shape[0] // tag_feat
    tags = tag_flat[: n_tags * tag_feat].reshape(n_tags, tag_feat)
    parsed = []
    for idx in range(n_tags):
        row = tags[idx]
        parsed.append(
            {
                "idx": idx,
                "center": (float(row[0]), float(row[1])),
                "corners": [
                    (float(row[2]), float(row[3])),
                    (float(row[4]), float(row[5])),
                    (float(row[6]), float(row[7])),
                    (float(row[8]), float(row[9])),
                ],
                "tag_id": float(row[10]),
                "visible": float(row[10]) >= 0.0,
            }
        )
    return parsed


def show_combined_scene_from_raw(raw_obs_flat, width: int, height: int):
    tag, img = _parse_tag_and_image(raw_obs_flat)
    if tag is None or img is None:
        return
    tags = _split_tags(tag)

    width = max(1, int(width))
    height = max(1, int(height))
    panel_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    panel_proj = np.zeros((height, width, 3), dtype=np.uint8)

    sx = float(width - 1) / 83.0
    sy = float(height - 1) / 83.0

    # Projection panel grid
    c_x = int(round(0.5 * float(width - 1)))
    c_y = int(round(0.5 * float(height - 1)))
    cv2.line(panel_proj, (0, c_y), (width - 1, c_y), (70, 70, 70), 1)
    cv2.line(panel_proj, (c_x, 0), (c_x, height - 1), (70, 70, 70), 1)

    def to_panel_xy(x84, y84):
        x = int(round(np.clip(float(x84) * sx, 0, width - 1)))
        y = int(round(np.clip(float(y84) * sy, 0, height - 1)))
        return x, y

    palette = [
        (0, 255, 255),
        (0, 200, 0),
        (255, 160, 0),
        (255, 0, 255),
        (255, 255, 0),
    ]
    visible_tags = [t for t in tags if t["visible"]]
    if visible_tags:
        for t in visible_tags:
            color = palette[t["idx"] % len(palette)]
            center = t["center"]
            corners = t["corners"]
            tid = int(round(t["tag_id"]))

            for i in range(4):
                p0 = to_panel_xy(*corners[i])
                p1 = to_panel_xy(*corners[(i + 1) % 4])
                cv2.line(panel_img, p0, p1, color, 2)
                cv2.circle(panel_img, p0, 4, color, -1)
            cpt = to_panel_xy(*center)
            cv2.circle(panel_img, cpt, 4, (0, 0, 255), -1)
            cv2.putText(panel_img, f"id{tid}", (cpt[0] + 6, cpt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            for i in range(4):
                p0 = to_panel_xy(*corners[i])
                p1 = to_panel_xy(*corners[(i + 1) % 4])
                cv2.line(panel_proj, p0, p1, color, 2)
                cv2.circle(panel_proj, p0, 4, color, -1)
            cv2.circle(panel_proj, cpt, 4, (0, 0, 255), -1)
            cv2.putText(panel_proj, f"id{tid} c=({center[0]:.1f},{center[1]:.1f})",
                        (max(0, cpt[0] - 30), max(15, cpt[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(panel_proj, f"visible tags: {len(visible_tags)}/{len(tags)}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    else:
        cv2.putText(panel_img, "no tag visible", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(panel_proj, "no tag visible", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(panel_img, "OBS IMAGE", (8, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel_proj, "PROJECTION", (8, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    scene = np.concatenate([panel_img, panel_proj], axis=1)
    cv2.imshow("Dot Scene (Image + Projection)", scene)
    cv2.waitKey(1)


def save_obs_image_from_raw(raw_obs_flat, save_path: str):
    _, img = _parse_tag_and_image(raw_obs_flat)
    if img is None:
        return False
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return bool(cv2.imwrite(save_path, img))


def get_raw_obs_from_vec_env(vec_env):
    base = vec_env
    while hasattr(base, "venv"):
        base = base.venv
    raw = getattr(base, "_observation", None)
    if raw is None:
        return None
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[0] < 1:
        return None
    return raw[0]


class ProjectionVizCallback(BaseCallback):
    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        show_windows: bool = True,
        show_scene: bool = False,
        save_obs_image: bool = False,
        obs_image_save_dir: str = "",
        obs_image_save_every: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.show_windows = bool(show_windows)
        self.show_scene = bool(show_scene)
        self.save_obs_image = bool(save_obs_image)
        self.obs_image_save_dir = obs_image_save_dir
        self.obs_image_save_every = max(1, int(obs_image_save_every))
        self._saved_count = 0

    def _on_step(self) -> bool:
        raw_obs = get_raw_obs_from_vec_env(self.training_env)
        if raw_obs is not None:
            if self.show_windows and self.show_scene:
                show_combined_scene_from_raw(raw_obs, self.width, self.height)
            if self.save_obs_image and (self.n_calls % self.obs_image_save_every == 0):
                save_path = os.path.join(
                    self.obs_image_save_dir,
                    f"train_step_{self.num_timesteps:09d}.png",
                )
                if save_obs_image_from_raw(raw_obs, save_path):
                    self._saved_count += 1
        return True


def main():
    args = parser().parse_args()
    print(f"[Debug] running script: {os.path.realpath(__file__)}")
    ensure_flightmare_path()
    apply_init_pos_override(args.init_pos)
    yaml = YAML()  # 기본 typ='rt' (RoundTrip)
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","..","flightlib/configs/vec_env.yaml"))
    
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f)

    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    need_unity_camera = bool(args.render) or bool(args.save_obs_image) or bool(args.viz_scene)
    if need_unity_camera and int(cfg["env"]["num_envs"]) > 8:
        print(
            f"[Warning] num_envs={cfg['env']['num_envs']} in Unity image mode may be unstable. "
            "Forcing num_envs=1, num_threads=1."
        )
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1
    cfg["env"]["render"] = "yes" if need_unity_camera else "no"

    # cfg를 YAML "문자열"로 다시 dump (QuadrotorEnv_v1이 이걸 받는 구조)
    stream = io.StringIO()
    yaml.dump(cfg, stream)
    cfg_yaml_str = stream.getvalue()

    # print(cfg_yaml_str)

    # main env
    use_obs_norm = bool(args.use_obs_norm)
    env = build_env(cfg_yaml_str, use_obs_norm=use_obs_norm)
    unity_connected = False
    if need_unity_camera:
        if not env.connectUnity():
            raise RuntimeError(
                "Failed to connect to Unity. Image observation/visualization requires Unity."
            )
        unity_connected = True

    # set random seed
    configure_random_seed(args.seed, env=env)
    # Log what the policy actually receives from the VecEnv wrapper.
    obs_preview = env.reset()
    obs_preview = np.asarray(obs_preview)
    print(f"[PolicyObs] shape={obs_preview.shape}, dtype={obs_preview.dtype}")
    if obs_preview.ndim >= 2 and obs_preview.shape[0] > 0:
        print(f"[PolicyObs] env0={obs_preview[0].tolist()}")

    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)

        n_envs = env.num_envs
        n_steps = 250
        batch_size = n_steps * n_envs  # emulate nminibatches=1
        checkpoint_freq_total = max(1, int(args.checkpoint_freq))
        checkpoint_freq_calls = max(1, checkpoint_freq_total // max(1, int(n_envs)))
        actual_checkpoint_freq_total = checkpoint_freq_calls * int(n_envs)
        print(
            f"[Train] checkpoint_freq(total)={checkpoint_freq_total}, "
            f"num_envs={n_envs} -> callback save_freq={checkpoint_freq_calls} "
            f"(actual total interval={actual_checkpoint_freq_total})"
        )

 

        # wandb init
        wandb_run = None
        if args.wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "algo": "PPO",
                    "seed": args.seed,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "n_steps": n_steps,
                    "batch_size": batch_size,
                    "n_epochs": 10,
                    "clip_range": 0.2,
                    "learning_rate": 3e-4,
                    "ent_coef": 0.0,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "num_envs": n_envs,
                },
                sync_tensorboard=True,  # SB3 TB 로그 자동 동기화
                monitor_gym=False,      # 우리는 VecMonitor를 이미 씀
                save_code=True,
            )

        model = PPO(
            policy="MlpPolicy",
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[dict(pi=[256, 256], vf=[512, 512])],
                log_std_init=-0.5,
            ),
            env=env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,          # PPO2 noptepochs
            gamma=0.99,
            gae_lambda=0.95,      # PPO2 lam
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=saver.data_dir,
            use_sde=False,
            verbose=1,
            device="cuda",
        )

        eval_env = None

        callback_list = []
        # NOTE:
        # Do not create eval_env in Unity image mode.
        # UnityBridge is a singleton and multiple VecEnv instances can corrupt
        # expected message layout (short/invalid image messages).
        if not need_unity_camera:
            cfg_eval = copy.deepcopy(cfg)
            cfg_eval["env"]["num_envs"] = 1
            cfg_eval["env"]["num_threads"] = 1
            cfg_eval["env"]["render"] = "no"
            stream_eval = io.StringIO()
            yaml.dump(cfg_eval, stream_eval)
            eval_env = build_env(stream_eval.getvalue(), use_obs_norm=use_obs_norm)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(saver.data_dir, "best_model"),
                log_path=os.path.join(saver.data_dir, "eval_logs"),
                eval_freq=max(1, int(args.eval_freq)),
                n_eval_episodes=max(1, int(args.n_eval_episodes)),
                deterministic=True,
                render=False,
                verbose=1,
                warn=False,
            )
            callback_list.append(eval_callback)
        else:
            print("[Train] Unity camera mode: EvalCallback(best model save) is disabled.")
        
        # Add observation normalization update callback only when enabled.
        # (Checkpoint callback itself is always enabled regardless of use_obs_norm.)
        if use_obs_norm:
            callback_list.append(ObsNormUpdateCallback())
            checkpoint_callback = CheckpointCallbackWithRMS(
                save_freq=checkpoint_freq_calls,
                save_path=os.path.join(saver.data_dir, "checkpoints"),
                name_prefix="ppo_model",
                verbose=1,
            )
            print("[Train] Checkpoint callback: CheckpointCallbackWithRMS")
        else:
            from stable_baselines3.common.callbacks import CheckpointCallback
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq_calls,
                save_path=os.path.join(saver.data_dir, "checkpoints"),
                name_prefix="ppo_model",
                verbose=1,
            )
            print("[Train] Checkpoint callback: CheckpointCallback")
        callback_list.append(checkpoint_callback)

        if args.viz_scene:
            callback_list.append(
                ProjectionVizCallback(
                    width=args.proj_img_width,
                    height=args.proj_img_height,
                    show_windows=True,
                    show_scene=True,
                    save_obs_image=bool(args.save_obs_image),
                    obs_image_save_dir=(
                        args.obs_image_save_dir
                        if args.obs_image_save_dir
                        else os.path.join(saver.data_dir, "obs_images")
                    ),
                    obs_image_save_every=args.obs_image_save_every,
                )
            )
        elif args.save_obs_image:
            callback_list.append(
                ProjectionVizCallback(
                    show_windows=False,
                    show_scene=False,
                    save_obs_image=True,
                    obs_image_save_dir=(
                        args.obs_image_save_dir
                        if args.obs_image_save_dir
                        else os.path.join(saver.data_dir, "obs_images")
                    ),
                    obs_image_save_every=args.obs_image_save_every,
                )
            )

        if args.wandb:
            callback_list.append(WandbCallback(
                gradient_save_freq=0,
                model_save_freq=100_000,
                model_save_path=os.path.join(saver.data_dir, f"wandb_{wandb_run.id}"),
                verbose=2,
            ))

        callbacks = CallbackList(callback_list)

        try:
            model.learn(
                total_timesteps=int(args.total_timesteps),
                tb_log_name="PPO_Flightmare",
                callback=callbacks,
                progress_bar=True,
            )

            model.save(os.path.join(saver.data_dir, "ppo_final"))
        finally:
            # 환경 정리
            if args.viz_scene:
                cv2.destroyAllWindows()
            if unity_connected:
                try:
                    env.disconnectUnity()
                except RuntimeError as e:
                    print(f"[Train] disconnectUnity warning: {e}")
            if eval_env is not None:
                eval_env.close()
            env.close()

    else:
        # Test mode (simple loop)
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),f'saved/{args.weight}/best_model/best_model.zip') 
        model = PPO.load(model_path, env=env, device="auto")
        
        # Load normalization statistics if normalization is enabled
        if use_obs_norm:
            rms_path = args.rms_path
            if rms_path is None:
                # Try to find RMS file from checkpoint directory
                checkpoint_dir = model_path
                rms_dir = os.path.join(checkpoint_dir, "RMS")
                if os.path.exists(rms_dir):
                    # Find the latest RMS file
                    rms_files = [f for f in os.listdir(rms_dir) if f.endswith('.npz')]
                    if rms_files:
                        # Sort by iteration number
                        rms_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                        rms_path = os.path.join(rms_dir, rms_files[-1])
                        print(f"[Test Mode] Found RMS file: {rms_path}")
                    else:
                        print(f"[Test Mode] Warning: No RMS files found in {rms_dir}")
                        print(f"[Test Mode] Continuing without normalization statistics (may affect performance)")
                else:
                    print(f"[Test Mode] Warning: RMS directory not found: {rms_dir}")
                    print(f"[Test Mode] Continuing without normalization statistics (may affect performance)")
            
            if rms_path and os.path.exists(rms_path):
                env.load_rms(rms_path)
                print(f"[Test Mode] Loaded normalization statistics from: {rms_path}")
            elif rms_path:
                print(f"[Test Mode] Warning: RMS file not found: {rms_path}")
                print(f"[Test Mode] Continuing without normalization statistics (may affect performance)")
        
        # Disable truncation for testing - allow episodes to run until crash or manual stop
        # This allows testing how long the model can hover without time limit
        env.wrapper.setTruncationEnabled(False)
        print(f"[Test Mode] Truncation disabled - episodes will run until crash or manual stop")
        
        max_ep_length = 1000  # Set a large limit for Python loop (C++ truncation is disabled)
        num_rollouts = 5

        for n_roll in range(num_rollouts):
            print(f"\n=== Rollout {n_roll} ===")

            # rollout buffers (optional)
            pixels, actions = [], []

            obs = env.reset()
            done = np.array([False])
            ep_len = 0

            while not (done[0] or ep_len >= max_ep_length):
                # policy inference
                act, _ = model.predict(obs, deterministic=True)

                # env step
                obs, reward, done, info = env.step(act)

                ep_len += 1

                # ---- logging (policy obs shape: [1, 8]) ----
                pixels.append(obs[0, 0:2].tolist())
                actions.append(act[0].tolist())

                if args.viz_scene:
                    raw_obs = get_raw_obs_from_vec_env(env)
                    if raw_obs is not None:
                        show_combined_scene_from_raw(
                            raw_obs,
                            width=max(1, int(args.proj_img_width)),
                            height=max(1, int(args.proj_img_height)),
                        )
                if args.save_obs_image:
                    raw_obs = get_raw_obs_from_vec_env(env)
                    if raw_obs is not None and (ep_len % max(1, int(args.obs_image_save_every)) == 0):
                        save_dir = (
                            args.obs_image_save_dir
                            if args.obs_image_save_dir
                            else os.path.join(args.save_dir, "obs_images_test")
                        )
                        save_path = os.path.join(
                            save_dir,
                            f"rollout_{n_roll:02d}_step_{ep_len:05d}.png",
                        )
                        save_obs_image_from_raw(raw_obs, save_path)

            print(f"Rollout {n_roll} finished | length = {ep_len}")

        if unity_connected:
            try:
                env.disconnectUnity()
            except RuntimeError as e:
                print(f"[Test] disconnectUnity warning: {e}")
        if args.viz_scene:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
