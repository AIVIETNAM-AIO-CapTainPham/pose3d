# 3D Pose Estimation

Dự án nghiên cứu và xây dựng hệ thống ước lượng tư thế con người trong không gian 3D từ ảnh 2D thời gian thực.

## Git Workflow

### Cấu trúc branch

```
main                        # tài liệu, ổn định
└── dev                     # tích hợp các feature
    └── feature/<tên>       # thêm tính năng mới
    └── fix/<tên>           # sửa bug
    └── chore/<tên>         # cập nhật config, docs,...
```

Ví dụ: `feature/data-preprocessing`, `fix/model-output-error`

### Các bước làm việc

```bash
# 1. Luôn cập nhật branch dev trước
git checkout dev
git pull origin dev

# 2. Tạo branch mới từ dev
git checkout -b feature/<tên-tính-năng>

# 3. Làm việc, sau đó commit
git add .
git commit -m "feat: mô tả ngắn thay đổi"

# 4. Push branch lên remote
git push origin feature/<tên-tính-năng>

# 5. Tạo Pull Request trên GitHub để merge vào dev
```

### Quy tắc commit message

| Prefix | Dùng khi |
|---|---|
| `feat:` | Thêm tính năng mới |
| `fix:` | Sửa bug |
| `docs:` | Cập nhật tài liệu |
| `chore:` | Thay đổi config, dependencies |
| `refactor:` | Refactor code |
