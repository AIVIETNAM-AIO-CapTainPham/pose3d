"""RTMPose3D-L finetune config — v4: validate mỗi epoch (không chỉ v3 + sửa).

Kế thừa toàn bộ v3 (regularization mạnh hơn, augmentation mạnh hơn, LR thấp +
decay sớm — xem docstring của rtmw3d_l_finetune_pose24_v3.py). Thay đổi duy
nhất ở đây: `val_interval` 5 → 1.

Vì sao: với val_interval=5, CheckpointHook(save_best="MPJPE") chỉ có thể chọn
"best" tại các epoch chia hết cho 5 (5, 10, 15...) — các epoch xen giữa
(31, 32, 36...) không được đánh giá nên không thể là best dù MPJPE ở đó có
thể thấp hơn. val_interval=1 loại bỏ hạn chế này, đổi lại train chậm hơn một
chút (val mỗi epoch) và log dài hơn.

Dùng:
    PYTHONPATH=src uv run python tools/train.py \
        src/pose24/configs/rtmw3d_l_finetune_pose24_v4.py \
        --work-dir work_dirs/pose24_v4 --amp
"""

_base_ = ["rtmw3d_l_finetune_pose24_v3.py"]

max_epochs = 100
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
