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
    np.random.seed(seed) #난수 생성 seed 설정 (numpy)
    torch.manual_seed(seed) #난수 생성 seed 설정(PyTorch)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed) #GPU에서의 난수 생성 seed 설정(PyTorch)

def ensure_flightmare_path(): #main()에서 호출 
    root_dir = os.path.abspath( #os.path.abspath : 절대경로로 변환 
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    )
    #os.path.realpath(__file__) : 현재 파일의 절대 경로를 반환
    # os.path.dirname() : 주어진 경로에서 디렉토리 부분을 반환. ..은 상위 디렉토리를 의미하므로, 두 번 사용하여 프로젝트 루트로 이동
    env_root = os.environ.get("FLIGHTMARE_PATH", "") #환경변수 FLIGHTMARE_PATH의 값을 가져오고, 없으면 빈 문자열 반환
    env_cfg = os.path.join(env_root, "flightlib", "configs", "quadrotor_env.yaml")
    if not env_root or not os.path.isfile(env_cfg): #env_root이 비어있거나, env_cfg 경로에 파일이 존재하지 않으면
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
    parser.add_argument('--wandb', type=int, default=0, help="Enable wandb logging")
    parser.add_argument('--wandb_project', type=str, default='flightmare_landing', help="wandb project name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="wandb run name")
    parser.add_argument('--wandb_episode_log_freq', type=int, default=10,
                        help="Log episode metrics to wandb every N episodes")
    parser.add_argument('--use_obs_norm', type=int, default=1, help="Use observation normalization (1=True, 0=False)")
    parser.add_argument('--include_prev_action', type=int, default=1,
                        help="Append previous action to policy observation (1=True, 0=False)")
    parser.add_argument('--rms_path', type=str, default=None, 
                        help="Path to normalization statistics (.npz file) for testing. "
                             "If None, will try to find RMS file from checkpoint directory.")
    
    #visualization and image saving 
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
    yaml = YAML() #raumel.yaml 라이브러리의 클래스 생성자 
    with open(quad_cfg_path, "r") as f: #r : 읽기 #open() : 파이썬 함수 파일 열기, with문 : 파일 자동 닫기
        quad_cfg = yaml.load(f)
    quad_cfg["quadrotor_env"]["init_pos"] = [float(init_pos[0]), float(init_pos[1]), float(init_pos[2])]
    with open(quad_cfg_path, "w") as f:
        yaml.dump(quad_cfg, f) #yaml.dump() : Python 객체를 YAML 형식으로 파일에 쓰기
        #quad_cfg 를 YAML 텍스트로 변환해서 f 에 써줌. f는 quad_cfg_path 파일을 가리키는 파일 객체(quadrotor_env.yaml)
    print(f"[Config] Overrode quadrotor_env.init_pos -> {quad_cfg['quadrotor_env']['init_pos']}")

def build_env(cfg_yaml_str, use_obs_norm=True, include_prev_action=True):
    env = wrapper.DotFlightEnvVec(   
        QuadrotorDotEnv_v1(cfg_yaml_str, False),
        use_obs_norm=use_obs_norm,
        include_prev_action=bool(include_prev_action),
    )
    env = VecMonitor(env)  # SB3 전용 래퍼. 에피소드 통계를 자동 기록. episode 끝날때  info 에 길이/리턴 같은 통계를 넣음. 이걸 Tensorboard 에서 집계함 
    # VecMonitor는 episode가 끝날 때마다 정보 업데이트. 
    return env
    #DotFlightEnvVec는 Stable Baselines3에서 사용할 수 있도록 Flightmare의 QuadrotorDotEnv_v1을 래핑한 클래스.
    #QuadrotorDotEnv_v1 는 Flightmare 시뮬레이터에서 제공하는 드론 제어 환경. pybind_wrapper.cpp 에서 C++로 구현된 환경을 Python에서 사용할 수 있도록 래핑한 클래스.
    # cfg_yaml_str은 환경 설정을 담은 YAML 문자열. use_obs_norm과 include_prev_action은 관측값 정규화와 이전 행동 포함 여부를 설정하는 플래그.


#raw image -> tag 정보와 이미지 정보 분리(_parse_tag_and_image) -> tag 정보에서 각 태그의 위치, ID, 가시성 등 추출 -> 이미지와 태그 정보를 시각화하는 함수들
def _parse_tag_and_image(raw_obs_flat):
    raw_obs_flat = np.asarray(raw_obs_flat, dtype=np.float32).reshape(-1) 
    # 84*84*3 = 21168, tag은 11차원. 
    #dytpe=np.float32 : 타입을 float32로 맞춤 
    #np.asarray() : 입력 데이터를 numpy 배열로 변환. 이미 numpy 배열이면 그대로 반환, 리스트나 다른 시퀀스면 배열로 변환. 
    # reshape(-1) : 1차원으로 평탄화. 전체 요소 수는 유지하면서 모든 차원을 하나로 합침.
    img_size = 84 * 84 * 3 
    if raw_obs_flat.shape[0] < (11 + img_size): #raw_obs_flat.shape == (21189,) 이면 raw_obs_flat.shap[0]= 21189
        return None, None #(tag, img) = (none, none)
    tag_dim = raw_obs_flat.shape[0] - img_size
    tag = raw_obs_flat[:tag_dim]
    img_flat = np.clip(raw_obs_flat[tag_dim:tag_dim + img_size], 0.0, 255.0).astype(np.uint8)
    #np.clip = 값들을 0-255 사이로 제한. 0보다 작은 값은 0으로, 255보다 큰 값은 255로 바꿈. uint8(8비트) 은 범위가 0-255 임
    img = img_flat.reshape(84, 84, 3)
    return tag, img


def _split_tags(tag_flat):
    tag_flat = np.asarray(tag_flat, dtype=np.float32).reshape(-1)
    tag_feat = 11
    if tag_flat.shape[0] < tag_feat:
        return []
    n_tags = tag_flat.shape[0] // tag_feat #태그 정보는 11차원씩 묶여있으므로, 전체 길이를 11로 나누면 태그 개수 나옴. 
    tags = tag_flat[: n_tags * tag_feat].reshape(n_tags, tag_feat) 
    #태그 정보 부분만 잘라서 (n_tags, 11) 형태로 재구성. 나머지 요소들은 이미지 정보이므로 무시.
    #tags = [[11개],[11개],[11개]]
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
                "tag_id": float(row[10]), #tag id. -1 이면 tag id 없는거고, 미검출임. 
                "visible": float(row[10]) >= 0.0, #visibility(tag_id 가 0보다 크면 1)
            }
        )
    return parsed


def show_combined_scene_from_raw(raw_obs_flat, width: int, height: int):
    tag, img = _parse_tag_and_image(raw_obs_flat) #raw_obs_flat에서 tag 정보와 이미지 정보를 분리해서 반환. tag는 11차원 벡터, img는 84x84x3 형태의 RGB 이미지로 변환.
    if tag is None or img is None:
        return
    tags = _split_tags(tag) #tag 별로 딕셔너리 형태로 만듬 

    width = max(1, int(width))
    height = max(1, int(height))
    panel_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    #cv2.INTER_NEARES = 이미지를 확대할 때 가장 가까운 픽셀의 값을 그대로 복사해서 확대하는 방식. 이미지가 픽셀화되어 보이지만, 태그 경계가 뚜렷하게 보이는 효과가 있음.
    panel_proj = np.zeros((height, width, 3), dtype=np.uint8)
    #panel_proj = [ [ [R,G,B] [] [] ] [            ] [] [] [] ...]

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
    while hasattr(base, "venv"): #hasattr : base 에 venv 라는 속성이 있는지 확인하는 파이썬 내장 함수 
        base = base.venv  #계속 래퍼를 벗겨낸다 
        #env = VecMonitor(DotFlightEnvVec(...)) 
        #바깥 래퍼는 안쪽 원본 env를 venv라는 이름으로 들고 있음
        #base = DotFlightEnvVec 
    raw = getattr(base, "_observation", None) #getattr : base 객체에서 _observation 이라는 속성을 가져옴. 없으면 None 반환.
    if raw is None:
        return None
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[0] < 1: #raw가 2차원 배열이 아니거나, 첫 번째 차원(environment 개수)의 크기가 1보다 작은 경우
        #raw = (num_envs, obs_dim)
        #raw = [[1,2,3,4...], [1,2,3,4,...], ...] 형태의 2차원 배열이어야 함.
        return None
    return raw[0] #첫번째 environment 의 raw observation 반환 


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
        raw_obs = get_raw_obs_from_vec_env(self.training_env) #첫번째 environment 의 observation 
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


class WandbExtraInfoCallback(BaseCallback):
    def __init__(self, env_index: int = 0, episode_log_freq: int = 1, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.env_index = max(0, int(env_index))
        self.episode_log_freq = max(1, int(episode_log_freq))
        self._episode_reward_sums = {}
        self._episode_env_return = 0.0
        self._episode_len = 0
        self._episode_count = 0

    def _on_step(self) -> bool:
        if wandb.run is None:
            return True

        infos = self.locals.get("infos", None)
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        if not infos or dones is None:
            return True
        if self.env_index >= len(infos): #env_index가 infos 리스트 길이보다 크거나 같으면, 현재 콜백에서 처리할 env가 없다는 의미이므로
            return True

        #infors -> info -> extra_info -> reward_로 시작하는 키들 -> episode_reward_sums에 누적
        info = infos[self.env_index]
        extra = info.get("extra_info", None) if isinstance(info, dict) else None
        if isinstance(extra, dict):
            for k, v in extra.items():
                if str(k).startswith("reward_"):
                    self._episode_reward_sums[k] = self._episode_reward_sums.get(k, 0.0) + float(v)

        if rewards is not None and len(rewards) > self.env_index:
            self._episode_env_return += float(rewards[self.env_index])
        self._episode_len += 1

        done_flag = bool(dones[self.env_index]) if len(dones) > self.env_index else False
        if done_flag:
            if self._episode_count % self.episode_log_freq == 0:
                log_data = {
                    "episode/env_return": float(self._episode_env_return),
                    "episode/length": int(self._episode_len),
                    "episode/index": int(self._episode_count),
                }
                for k, v in self._episode_reward_sums.items():
                    log_data[f"episode/{k}_sum"] = float(v)
                wandb.log(log_data, step=self.num_timesteps)
            self._episode_reward_sums = {}
            self._episode_env_return = 0.0
            self._episode_len = 0
            self._episode_count += 1
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
    include_prev_action = bool(args.include_prev_action)
    env = build_env(
        cfg_yaml_str,
        use_obs_norm=use_obs_norm,
        include_prev_action=include_prev_action,
    )
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
            eval_env = build_env(
                stream_eval.getvalue(),
                use_obs_norm=use_obs_norm,
                include_prev_action=include_prev_action,
            )
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
            callback_list.append(
                WandbExtraInfoCallback(
                    env_index=0,
                    episode_log_freq=max(1, int(args.wandb_episode_log_freq)),
                    verbose=0,
                )
            )
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
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),f'saved/{args.weight}/checkpoints/ppo_model_100000000_steps.zip') 
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
            total_reward = 0

            while not (done[0] or ep_len >= max_ep_length):
                # policy inference
                act, _ = model.predict(obs, deterministic=True)
                print(f"step {ep_len:04d} | obs0={np.asarray(obs[0], dtype=np.float32)} | act0={act[0].tolist()}")

                # env step
                obs, reward, done, info = env.step(act)
                total_reward += reward[0]
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

            print(f"Rollout {n_roll} finished | length = {ep_len} | reward = {total_reward :.2f}")

        if unity_connected:
            try:
                env.disconnectUnity()
            except RuntimeError as e:
                print(f"[Test] disconnectUnity warning: {e}")
        if args.viz_scene:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
