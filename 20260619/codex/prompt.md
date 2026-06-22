# Codex 実装指示: G1右腕をかご上空へ移動してグリッパーを開くプログラム

## 目的

Unitree G1ヒューマノイドロボットの右腕を、任意の現在位置から「左手のかご上空」の固定姿勢へ補間移動し、到達後に右グリッパーを開いてオクラをリリースするPythonスクリプトを作成する。

**まずMuJoCoシミュレーション上で動作確認できるものを作る。** 実機用のDDS通信版は後で作る。

## 成果物

`20260617/オクラをかごの中に入れるタスク/src/` 配下に以下を作成:

1. `drop_to_basket_mujoco.py` — MuJoCoシミュレーション上でG1右腕を投入姿勢へ移動しグリッパーを開くデモ
2. `arm_interpolator.py` — 関節角度の補間ロジック（MuJoCo版・実機版で共用）

## 環境情報

### MuJoCo

- MuJoCoワークスペース: `/home/techshare/drl_kit/mujoco_ws/ts_mujoco-main/`
- G1モデル（手付き）: `/home/techshare/drl_kit/mujoco_ws/ts_mujoco-main/unitree_robots/g1/scene_29dof_with_hand.xml`
- G1モデル本体: `/home/techshare/drl_kit/mujoco_ws/ts_mujoco-main/unitree_robots/g1/g1_29dof_with_hand.xml`
- 参考: 既存のMuJoCo起動スクリプト `/home/techshare/drl_kit/mujoco_ws/ts_mujoco-main/simulate_python/unitree_mujoco.py`
- MuJoCoのPythonパッケージ `mujoco` のインストール要否を確認し、なければインストールすること

### G1の右腕関節（MuJoCoモデル上のjoint name）

制御対象の右腕7関節:

| インデックス | joint name | 説明 |
|---|---|---|
| 0 | right_shoulder_pitch_joint | 肩ピッチ |
| 1 | right_shoulder_roll_joint | 肩ロール |
| 2 | right_shoulder_yaw_joint | 肩ヨー |
| 3 | right_elbow_joint | 肘 |
| 4 | right_wrist_roll_joint | 手首ロール |
| 5 | right_wrist_pitch_joint | 手首ピッチ |
| 6 | right_wrist_yaw_joint | 手首ヨー |

各関節の可動範囲（URDFより、単位: rad）:

| joint | lower | upper |
|---|---|---|
| right_shoulder_pitch | -3.0892 | 2.6704 |
| right_shoulder_roll | -2.2515 | 1.5882 |
| right_shoulder_yaw | -2.618 | 2.618 |
| right_elbow | -1.0472 | 2.0944 |
| right_wrist_roll | -1.9722 | 1.9722 |
| right_wrist_pitch | -1.6144 | 1.6144 |
| right_wrist_yaw | -1.6144 | 1.6144 |

### 左腕（かご側、制御しないがホールドが必要）

左腕7関節は起動姿勢を維持する（kpホールド）。制御対象ではないが、MuJoCo上でも脱力しないよう固定値を送り続けること。

| joint name |
|---|
| left_shoulder_pitch_joint |
| left_shoulder_roll_joint |
| left_shoulder_yaw_joint |
| left_elbow_joint |
| left_wrist_roll_joint |
| left_wrist_pitch_joint |
| left_wrist_yaw_joint |

### 腰（制御しないがホールドが必要）

| joint name |
|---|
| waist_yaw_joint |
| waist_roll_joint |
| waist_pitch_joint |

## 実装仕様

### drop_to_basket_mujoco.py

1. MuJoCoでG1モデル（`scene_29dof_with_hand.xml`）を読み込みビューワーを起動
2. G1を立位姿勢で安定させる（脚関節は初期姿勢を保持）
3. 右腕をランダムな開始姿勢（可動範囲内）に配置（収穫後の様々な位置を模擬）
4. 1秒待機後、右腕を「投入姿勢」へ補間移動（1〜2秒かけて）
5. 到達後、右グリッパーを開く動作を模擬
6. 動作完了後もビューワーは開いたままにする

### 投入姿勢（仮の値、後で実機計測値に差し替える）

右手が左手のかご上空に来る姿勢を仮定義する。MuJoCo上で目視確認しながら調整できるよう、スクリプト先頭に定数として定義:

```python
# 投入姿勢（仮値、後で実機計測値に差し替え）
# 右手が体の左前方・かご上空に来る姿勢
DROP_POSE = {
    "right_shoulder_pitch_joint": 0.0,   # 後で調整
    "right_shoulder_roll_joint": 0.3,    # 後で調整
    "right_shoulder_yaw_joint": 0.0,     # 後で調整
    "right_elbow_joint": 1.0,            # 後で調整
    "right_wrist_roll_joint": 0.0,       # 後で調整
    "right_wrist_pitch_joint": 0.0,      # 後で調整
    "right_wrist_yaw_joint": 0.0,        # 後で調整
}
```

### arm_interpolator.py

関節角度の補間ユーティリティ:

```python
class ArmInterpolator:
    def __init__(self, start_angles, goal_angles, duration_s, control_freq_hz=250):
        """
        start_angles: dict[str, float] — 開始関節角度
        goal_angles: dict[str, float] — 目標関節角度
        duration_s: float — 移動にかける時間（秒）
        control_freq_hz: int — 制御周波数
        """
    
    def get_target(self, step: int) -> dict[str, float]:
        """step番目の補間目標角度を返す"""
    
    def is_done(self, step: int) -> bool:
        """補間完了かどうか"""
    
    @property
    def total_steps(self) -> int:
        """総ステップ数"""
```

補間方式は線形補間でよい:
```
target[i] = start + (goal - start) * (step / total_steps)
```

### MuJoCoでの制御方法

MuJoCoでは `mj_data.ctrl` にアクチュエータへの指令値を書き込み、`mj_step()` でシミュレーションを進める。

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("path/to/scene_29dof_with_hand.xml")
data = mujoco.MjData(model)

# joint名からアクチュエータIDを取得
actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch")

# 制御値を設定
data.ctrl[actuator_id] = target_angle

# シミュレーションを1ステップ進める
mujoco.mj_step(model, data)
```

## 注意事項

- MuJoCoのG1モデル内のアクチュエータ名はjoint名と異なる場合がある（`_joint` が付かない等）。モデルを読み込んだ後に `mj_name2id` で確認すること
- 脚関節は直接制御しない。MuJoCoの初期qpos（モデルに定義された立位姿勢）をそのまま保持する
- コード内のコメントは最小限にし、定数定義は分かりやすい変数名で表現する
- Python 3.10以上を想定
