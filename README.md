# Pose Image Scraper

Scrape ảnh pose người từ **Pinterest** và **Google Images**, lọc trùng bằng **Blake2b**, chỉ giữ lại ảnh có người chiếm **≥ 1/3 diện tích ảnh** (YOLOv11).

---

## Cấu trúc project

```
scrape_data/
├── pyproject.toml              # cấu hình uv + danh sách thư viện
├── config.yaml                 # cài đặt platform, đường dẫn, ngưỡng filter
├── keywords.yaml               # danh sách keyword cần scrape
├── convert_cookies.py          # convert cookies JSON → Netscape txt
├── cookies.txt                 # file cookies Pinterest (không commit)
├── .gitignore
└── scrape_data/
    ├── main.py                 # điểm vào CLI
    ├── dedup.py                # dedup Blake2b + SQLite
    ├── person_filter.py        # kiểm tra người với YOLOv11
    └── scrapers/
        ├── pinterest.py        # scraper dùng gallery-dl
        └── google.py           # scraper HTTP/2 httpx + fallback DuckDuckGo
```

---

## Cài đặt

### 1. Cài uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Tạo môi trường và cài thư viện

```bash
cd scrape_data
uv sync
```

Tự tạo `.venv/` và cài hết thư viện trong `pyproject.toml`.

---

## Lấy cookies Pinterest

Pinterest yêu cầu đăng nhập qua cookies. Làm theo các bước sau:

### Bước 1 — Cài extension J2TEAM Cookies

Vào link sau và click **Thêm vào Chrome**:

> [J2TEAM Cookies — Chrome Web Store](https://chromewebstore.google.com/detail/j2team-cookies/okpidcojinmlaakglciglbpcpajaibco)

### Bước 2 — Pin extension lên thanh toolbar

Sau khi cài xong:
1. Click icon **puzzle piece** (Extensions) góc phải trình duyệt
2. Tìm **J2TEAM Cookies** → click icon **pin** để ghim lên toolbar

### Bước 3 — Đăng nhập Pinterest

Vào [Pinterest](https://www.pinterest.com) và đăng nhập tài khoản bình thường.

### Bước 4 — Export cookies ra JSON

1. Khi đang ở trang Pinterest (đã đăng nhập), click icon **J2TEAM Cookies** trên toolbar
2. Click **Export Cookies**
3. File `pinterest.com.json` (hoặc tên tương tự) sẽ tải về máy
4. Copy file đó vào thư mục `scrape_data/` (cùng chỗ với `convert_cookies.py`)

### Bước 5 — Convert JSON → Netscape txt

`gallery-dl` cần cookies theo định dạng Netscape, không phải JSON. Chạy:

```bash
uv run python convert_cookies.py pinterest.com.json cookies.txt
```

Output:
```
Convert xong: 24 cookies
  Input  : pinterest.com.json
  Output : cookies.txt
```

File `cookies.txt` đã sẵn sàng. File này được `.gitignore` nên không bị commit lên git.

---

## Cấu hình

### `config.yaml` — platform & filter

| Key | Mặc định | Mô tả |
|-----|----------|-------|
| `platforms.pinterest.enabled` | `true` | Bật/tắt scrape Pinterest |
| `platforms.pinterest.max_per_keyword` | `0` | Số ảnh tối đa/keyword (`0` = không giới hạn) |
| `platforms.google.enabled` | `true` | Bật/tắt scrape Google |
| `platforms.google.max_per_keyword` | `200` | Số ảnh tối đa/keyword |
| `platforms.google.image_filter.size` | `large` | `large` / `medium` / `icon` |
| `platforms.google.image_filter.image_type` | `photo` | `photo` / `clipart` / `animated` |
| `person_filter.model` | `yolo11x.pt` | Tên hoặc đường dẫn model YOLO |
| `person_filter.confidence` | `0.4` | Ngưỡng confidence YOLO |
| `person_filter.min_area_ratio` | `0.333` | Người phải chiếm ≥ tỉ lệ này so với ảnh |
| `paths.cookies` | `./cookies.txt` | File cookies Pinterest |
| `paths.output_dir` | `./data/images` | Thư mục lưu ảnh đã duyệt |
| `paths.hash_db` | `./data/hashes.db` | Database dedup Blake2b |

### `keywords.yaml` — keyword cần scrape

```yaml
keywords:
  - "pose on cafe"
  - "pose on stairs"
  - "pose on park"
```

Mỗi keyword tạo thành 1 thư mục con trong `data/images/`. Thêm/xóa keyword tại đây, không cần động vào `config.yaml`.

---

## Chạy

### Lần đầu — tải model YOLOv11

Model tự tải về khi chạy lần đầu. Muốn tải trước:

```bash
uv run python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
```

### Chạy scrape

```bash
# Toàn bộ (tất cả platform, tất cả keyword)
uv run scrape

# Chỉ Pinterest
uv run scrape --platform pinterest

# Chỉ Google
uv run scrape --platform google

# Chỉ 1 keyword
uv run scrape --keyword "pose on cafe"

# Dùng file config khác
uv run scrape --config my_config.yaml
```

---

## Flow xử lý từng ảnh

```
┌─────────────────────────────────────────────────────────┐
│  Scraper tải ảnh về thư mục temp                        │
│  (Pinterest: gallery-dl  /  Google: httpx HTTP/2)       │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  Blake2b dedup                                          │
│                                                         │
│  1. Đọc raw bytes của ảnh                               │
│  2. Tính hash: blake2b(bytes, digest_size=20) → hex     │
│  3. Tra cứu hash trong SQLite (hashes.db)               │
│     • Có rồi  → bỏ qua, xóa file temp        [SKIP]    │
│     • Chưa có → tiếp tục bước tiếp theo       [OK]     │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  YOLOv11 — kiểm tra người                               │
│                                                         │
│  1. Detect tất cả bounding box class "person"           │
│  2. Lấy bbox lớn nhất                                   │
│  3. Tính: diện_tích_bbox / diện_tích_ảnh               │
│     • Không có người      → bỏ qua           [SKIP]    │
│     • Tỉ lệ < 1/3 (0.333) → người quá nhỏ   [SKIP]    │
│     • Tỉ lệ ≥ 1/3         → đạt yêu cầu      [OK]     │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  Lưu ảnh                                                │
│                                                         │
│  1. Ghi hash vào SQLite (tránh trùng lần sau)           │
│  2. Move file → data/images/{keyword}/filename.jpg      │
└─────────────────────────────────────────────────────────┘
```

---

## Kết quả đầu ra

```
data/
└── images/
    ├── pose_on_cafe/
    │   ├── pinterest_abc123.jpg
    │   └── google_def456.jpg
    ├── pose_on_stairs/
    │   └── ...
    └── ...
logs/
└── scrape.log
```

Thống kê cuối mỗi lần chạy:

```
==================================================
  RESULTS
==================================================
  Scraped     : 1,240
  Duplicate   : 87
  No person   : 503
  Saved       : 650  (52.4%)
  Hash DB     : 1,240 entries
  Output dir  : /path/to/data/images
==================================================
```
