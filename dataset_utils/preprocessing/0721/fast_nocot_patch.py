"""
no-CoT 预处理加速补丁（opt-in，纯 monkeypatch，不改任何原文件）。

为什么需要
----------
`nocot_sample_generation.py` 最终只把这些字段写进 JSON：
    token / velocity / acceleration / instruction / gt_trajectory / his_trajectory
    / <cam>_camera_paths
但产生它们的路上做了三件在 no-CoT 下**完全被丢弃**的重活：

  1. `Cameras.from_camera_dict` 在填 `camera_path` 的同一行做
     `image=np.array(Image.open(path))`。而 VlaAgent.get_sensor_config() 是
     8 个相机全开 × num_history_frames=4 => **每个场景解码 32 张 1920x1080 JPEG**。
     103,288 个场景 = 约 330 万次解码。JSON 只需要那个路径字符串。

  2. `nuplan_dataset.process_image_input` 对其中 16 张做
     cvtColor + imencode + base64。

  3. `process_vision_info` 把这 16 张 base64 解回来再缩放到 400x400。

三步产出的 `text` / `image_inputs` / `video_inputs` 被
`nocot_sample_generation.process_sample()` 原样丢弃。

补丁做什么
----------
  1. `Cameras.from_camera_dict` -> 只填 camera_path，不 Image.open
  2. `process_image_input`      -> 返回空串
  3. `process_vision_info`      -> 返回 (None, None)

`camera_path` 取自同一个 metadata dict（camera_dict[name]["data_path"]），
与原实现逐字符一致，所以产出的 JSON 应当**完全相同**。

⚠️ 只能用于 no-CoT 路径。CoT 标注需要真实图像，打了这个补丁会拿到空图。

用法
----
    from dataset_utils.preprocessing.fast_nocot_patch import apply
    apply()          # 在构造 dataset 之前调用
"""
import os
from pathlib import Path

_applied = False
_orig = {}


CHECK_EXISTS = True   # 用 os.path.exists 替代 Image.open，保留"图缺了要报错"的检查


def apply(check_exists: bool = True):
    """打上补丁（幂等）。

    :param check_exists: 是否仍然核对图片文件存在。原实现靠 Image.open 隐式
        保证了这一点；补丁不读图就丢掉了这道检查，缺图会拖到训练时才炸。
        改用 os.path.exists 只多约 5% 开销（32 次 stat/场景），强烈建议保持 True。
    """
    global _applied, CHECK_EXISTS
    if _applied:
        return
    CHECK_EXISTS = check_exists
    _apply_camera_patch()
    _apply_vision_patches()
    _apply_sensor_config_patch()
    _applied = True
    print(f"[fast_nocot_patch] 已启用：跳过图像解码/base64/vision 预处理"
          f"（文件存在性检查：{'开' if check_exists else '关'}）")


def revert():
    """还原（主要给测试用）。"""
    global _applied
    if not _applied:
        return
    from navsim.common import dataclasses as nsdc
    from dataset_utils.preprocessing import nuplan_dataset as npd

    from navsim.agents import vla_agent as va

    nsdc.Cameras.from_camera_dict = _orig["from_camera_dict"]
    npd.process_image_input = _orig["process_image_input"]
    npd.process_vision_info = _orig["process_vision_info"]
    va.VlaAgent.get_sensor_config = _orig["get_sensor_config"]
    _applied = False
    print("[fast_nocot_patch] 已还原")


def _apply_camera_patch():
    from navsim.common import dataclasses as nsdc

    _orig["from_camera_dict"] = nsdc.Cameras.from_camera_dict

    _CAM_FIELDS = ("cam_f0", "cam_l0", "cam_l1", "cam_l2",
                   "cam_r0", "cam_r1", "cam_r2", "cam_b0")

    @classmethod
    def paths_only(cls, sensor_blobs_path: Path, camera_dict, sensor_names):
        """与原实现同构，唯一区别：image 不解码，保持 None。"""
        data = {}
        for camera_name in camera_dict.keys():
            cid = camera_name.lower()
            if cid in sensor_names:
                image_path = sensor_blobs_path / camera_dict[camera_name]["data_path"]
                if CHECK_EXISTS and not os.path.exists(image_path):
                    # 保留原实现靠 Image.open 隐式提供的完整性检查
                    raise FileNotFoundError(f"缺少图像文件: {image_path}")
                data[cid] = nsdc.Camera(
                    image=None,                                   # ← 唯一改动
                    sensor2lidar_rotation=camera_dict[camera_name]["sensor2lidar_rotation"],
                    sensor2lidar_translation=camera_dict[camera_name]["sensor2lidar_translation"],
                    intrinsics=camera_dict[camera_name]["cam_intrinsic"],
                    distortion=camera_dict[camera_name]["distortion"],
                    camera_path=str(image_path),
                )
            else:
                data[cid] = nsdc.Camera()
        return nsdc.Cameras(**{f: data[f] for f in _CAM_FIELDS})

    nsdc.Cameras.from_camera_dict = paths_only


def _apply_vision_patches():
    from dataset_utils.preprocessing import nuplan_dataset as npd

    _orig["process_image_input"] = npd.process_image_input
    _orig["process_vision_info"] = npd.process_vision_info

    npd.process_image_input = lambda image: ""          # 不做 cvtColor/imencode/base64
    npd.process_vision_info = lambda messages: (None, None)   # 不解码不缩放


NUM_HISTORY_FRAMES = 4


def _apply_sensor_config_patch():
    """只加载 history+current 帧的相机，不碰未来 10 帧。

    为什么必须这么做
    ----------------
    `Scene.from_scene_dict_list` 对 **全部 14 帧** 逐帧调 `Cameras.from_camera_dict`
    (navsim/common/dataclasses.py:436)，而 `VlaAgent.get_sensor_config()` 把 8 个相机
    都设成 `True`（bool => 所有帧都算命中）。于是每个场景要碰 14x8 = 112 张图。

    但 `Scene.get_agent_input()` 只取 `range(num_history_frames)` 前 4 帧
    (dataclasses.py:355)，未来 10 帧的图**从来不会进入模型**，纯属白读。

    更要命的是：navtrain 的 445GB sensor 包（navtrain_current_* + navtrain_history_*）
    **只含 history+current 帧**，未来帧的 jpg 根本不存在 =>
    不做这个限制就会 FileNotFoundError。AutoVLA 作者用的是完整 trainval(2TB)，
    所有帧都在，所以上游没暴露这个问题。

    SensorConfig 的字段是 Union[bool, List[int]]，传帧号列表即可精确限制。
    """
    from navsim.agents import vla_agent as va
    from navsim.common.dataclasses import SensorConfig

    _orig["get_sensor_config"] = va.VlaAgent.get_sensor_config
    hist = list(range(NUM_HISTORY_FRAMES))

    def limited(self):
        return SensorConfig(cam_f0=hist, cam_l0=hist, cam_l1=hist, cam_l2=hist,
                            cam_r0=hist, cam_r1=hist, cam_r2=hist, cam_b0=hist,
                            lidar_pc=False)

    va.VlaAgent.get_sensor_config = limited
