# Ghi Chu Quan Ly

## Trang thai git/worktree

Luc ra soat, `git status --short` cho thay:

```text
 M .gitignore
?? data/
?? vis/
?? work_dirs/
```

Nghia la data, visualization va checkpoint/log dang la untracked trong git.
Thu muc `project_markdown/` moi duoc tao them de quan ly tai lieu.

## Phan nen xem la source of truth

Source code chinh:

- `src/pose24`
- `tools`
- `demo/app.py`
- `tests`
- `pyproject.toml`
- `Makefile`
- config tai `src/pose24/configs/rtmw3d_l_finetune_pose24.py`

Data runtime chinh:

- `data/GT/images`
- `data/GT/labels`
- `data/GT/splits`

Artifacts:

- `work_dirs`
- `vis`

Reference/vendor:

- `clone_code/pose3d`
- `clone_code/theia`

## Rui ro va diem can chu y

1. README da duoc cap nhat de dung `demo/app.py` va moi truong build `mmcv`
   theo stack hien tai `torch cu130` + `/usr/local/cuda-13.0`.

2. Dung luong data bi nhan doi:
   `data/GT/images` va `data/GT/combined/images` cung 18,752 file va cung
   6,485.6 MiB. Can quy uoc ro `combined` la backup/staging hay duplicate co
   the xoa/khong sync.

3. `labels_backup` cung 18,752 file. Nen co README nho trong `data/GT` neu day
   la ban truoc/sau fix truc Y.

4. Run `work_dirs/pose24` cho thay metric tot nhat quanh epoch 30, sau do degrade
   den epoch 200. Khi chon checkpoint cho demo/eval, nen uu tien best checkpoint
   thay vi last checkpoint.

5. [DA FIX] Camera focal length: tung dung 1 fallback hard-code chung cho moi
   sample (`[2074.0,2074.0]`, median cua dataset). GT JSON co `focal_length_px`
   rieng cho tung sample (phan phoi rat rong: ~775-6900px, std~764) -- nhung
   day la gia tri SAM-3D-Body UOC LUONG luc tao pseudo-GT (`pose3d_source:
   "sam3d"`), khong phai thong so camera vat ly do duoc; RTMPose3D khong tu
   suy luan ra f, chi dung nhu hang so co san. `GTJsonDataset` truoc do chua
   doc field nay -> sample co f uoc luong lech xa median bi outlier MPJPE
   (vi du 1349mm) do back-projection sai scale, khong phai do model du doan
   sai. Da sua: `GTJsonDataset._parse_json` doc `focal_length_px` +
   `image_size`, dong goi thanh `camera_param` moi sample ->
   `TopdownPoseEstimator3D` dung dung f (uoc luong tu SAM-3D-Body) cho tung
   anh thay vi 1 gia tri chung. Verify: outlier
   1349mm -> 50.5mm. Fallback `[2074.0,2074.0]` gio chi con dung cho anh
   khong co GT (anh upload trong demo). Chi tiet: ARCHITECTURE.md SS10.9.

5b. [DA FIX] Nhanh fallback con dao thu tu `(cx,cy)`: `c =
    ori_shape/2` nhung `ori_shape` la `(H,W)` theo convention mmpose, can
    `c=(W/2,H/2)` -> bi dao thanh `(cy,cx)`. Vo hai tren anh vuong nhung sai
    hang tram px tren anh khong vuong (96% dataset nay khong vuong). Da sua
    trong `pose_estimator.py`. Chi tiet: ARCHITECTURE.md SS10.10.

6. Dataset hien set `keypoints_visible` all-one. Neu label co visibility/occlusion
   that, nen dua vao JSON va dung trong dataset/loss.

7. `torch.load(weights_only=False)` dang dung cho checkpoint local tin cay. Khong
   nen dung checkpoint khong ro nguon voi setting nay.

8. Build `mmcv==2.2.0` tren GB10/aarch64 la build source, khong phai chi tai
   wheel. Lan verify gan nhat build xong trong khoang 32 phut voi `MAX_JOBS=8`.
   Neu may bi cham/swap, giam `MAX_JOBS=4`.

9. `albumentations` phai pin dung `==1.3.1` (version 2.x khong tuong thich voi
   `Albumentation` transform cua mmpose). Da gap truong hop package nay bi
   roi khoi venv sau khi chay lenh cai dat khac (`uv add`/`uv sync` tu thao
   tac khong lien quan) — `train-resume` crash ngay khi build dataset voi loi
   `RuntimeError: albumentations is not installed`. Cach fix:
   `uv pip install 'albumentations==1.3.1'` roi `make train-resume` lai (khong
   mat checkpoint vi crash xay ra truoc khi train iteration nao moi).

10. **Khong chay demo Streamlit (`make demo`) cung luc voi training.** Da co
    1 lan server sap, nghi do OOM: training RTMPose-L + demo nap 3 model
    (finetuned + original 133-kpt + dataset) cung luc len GPU/RAM. Train xong
    moi mo demo, hoac giam batch_size neu can chay song song.

## De xuat quan ly thu muc

Co the them README nho trong:

- `data/GT/README.md`: giai thich `labels`, `labels_backup`, `combined`.
- `work_dirs/README.md`: bang checkpoint nao nen dung, run nao la thu nghiem.
- `clone_code/README.md`: noi ro day la repo tham chieu, khong sua truc tiep.

## De xuat versioning/checkpoint

Nen tao bang checkpoint chuan, vi hien co nhieu checkpoint:

| Alias de xuat | Path | Ghi chu |
|---|---|---|
| `best_pose24_epoch30` | `work_dirs/pose24/best_MPJPE_epoch_30.pth` | best log run dai |
| `last_pose24_epoch200` | `work_dirs/pose24/epoch_200.pth` | last run dai, metric degrade |
| `best_pose24_v2_epoch15` | `work_dirs/pose24_v2/best_MPJPE_epoch_15.pth` | run v2 ngan |
| `original_rtmw3d_l` | `work_dirs/original_rtmw3d/rtmw3d-l_cocktail14.pth` | model goc |

## Viec nen lam tiep

- Neu thay doi torch/CUDA, cap nhat lai README phan build `mmcv` de tranh mismatch.
- Them tai lieu data provenance: data tu dau, `labels_backup` la truoc/sau buoc nao.
- Chay lai `make test` sau moi thay doi code/model pipeline.
- Chon 1 checkpoint mac dinh cho demo va ghi ro trong app/README.
- Xem xet clean hoac di chuyen duplicate data sang storage ngoai git repo.

