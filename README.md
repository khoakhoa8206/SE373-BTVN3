# SE373 · BTVN#3 · Agent đặt vé máy bay bằng LangChain

Bài tập về nhà buổi 03, môn SE373 *Kỹ thuật xây dựng hệ thống Agentic AI* (UIT).

Đề bài (slide 69): tìm hiểu LangChain, LangGraph → tạo tool mockup → viết lớp harness cho agent đặt vé máy bay.

1. Đủ các lớp: ràng buộc là dữ liệu, tiêu chí hoàn thành kiểm bằng code, kiểm quyền, bàn giao.
2. Cài đặt agent với 3 mẫu thiết kế: ReAct, Plan-then-Execute, Lai.
3. Đánh giá hiệu quả của agent với 3 mẫu thiết kế.

Yêu cầu người dùng dùng trong bài:

> Đặt giúp tôi vé SGN → DAD ngày 07/10/2026, cất cánh trước 12:00, giá dưới 2.000.000đ.

Báo cáo đầy đủ: [`BAO_CAO_BTVN3.docx`](BAO_CAO_BTVN3.docx).

## Cấu trúc

| File | Nội dung |
|---|---|
| [`btvn3_dat_ve.py`](btvn3_dat_ve.py) | Tool mockup, harness, 3 mẫu thiết kế, model giả lập, đánh giá |
| [`lib/harness.py`](lib/harness.py) | `LoopDetector`, `ban_giao`, `chan_truoc_khi_tra_loi`, lấy từ demo buổi 03 |
| [`test_btvn3.py`](test_btvn3.py) | 27 test bằng `unittest` |
| [`demo_btvn3.html`](demo_btvn3.html) | Trang xem trace từng vòng và bảng đánh giá, mở trực tiếp bằng trình duyệt |
| [`BAO_CAO_BTVN3.docx`](BAO_CAO_BTVN3.docx) | Báo cáo Word |

## Cài đặt và chạy

Cần Python 3.10 trở lên. Không cần API key: mặc định agent chạy bằng model giả lập.

```bash
pip install -r requirements.txt

python btvn3_dat_ve.py --mau react --kich-ban chuan    # in trace một lượt chạy
python btvn3_dat_ve.py --danh-gia                      # 5 kịch bản × 3 mẫu, in bảng so sánh
python -m unittest -v test_btvn3                       # chạy test
```

| Tham số | Giá trị |
|---|---|
| `--mau` | `react`, `plan`, `lai` |
| `--kich-ban` | `chuan`, `doi_gia`, `can_duyet`, `loi_dich_vu`, `ngan_sach` |
| `--that` | Dùng model thật qua `init_chat_model`. Cần đặt `SE373_MODEL` (ví dụ `anthropic:claude-sonnet-5-5`), có API key và cài gói của nhà cung cấp. |

Trên Windows, nếu terminal hiện sai tiếng Việt thì đặt `PYTHONIOENCODING=utf-8` trước khi chạy.

## Harness

Cả ba mẫu đi qua cùng một lớp `Harness`. Mọi lời gọi tool đều phải qua `Harness.thuc_thi()`.

| Lớp | Cách làm | Chạy khi nào |
|---|---|---|
| Ràng buộc là dữ liệu | `RangBuoc` (ngày, giờ, trần giá), so với giá mới nhất từ `check_seat` | Trước `book_seat`; vi phạm thì trả `denied` để model tự sửa |
| Tiêu chí hoàn thành | `get_booking` phải `confirmed`, đã trả tiền, không vi phạm ràng buộc | Sau mỗi observation; model tự nói "xong" vẫn bị kiểm lại |
| Kiểm quyền | Vé không hoàn hoặc giá trên 1.900.000đ thì dừng chờ duyệt; `pay` chỉ nhận mã từ `book_seat`; tool lạ bị chặn | Trước khi thực thi tool |
| Phát hiện lặp, bế tắc | `LoopDetector` so cặp `(tool, args)` và đo tiến triển theo chặng | Trước khi thực thi tool |
| Ngân sách | Số lần gọi model (mặc định 12), cộng `recursion_limit` của LangGraph | Kiểm sau cùng |
| Bàn giao | Lý do dừng, các bước đã thử, trạng thái và tác dụng phụ, câu hỏi cho người | Mỗi lần dừng bất thường |

## Ba mẫu thiết kế

- **ReAct**: `create_agent` cộng `HarnessMiddleware` (`before_model`, `after_model`, `wrap_tool_call`). Model chọn từng bước.
- **Plan-then-Execute**: `StateGraph` gồm node `lap_ke_hoach` và `thuc_thi`. Lập kế hoạch một lần rồi làm từng bước; bước nào hỏng thì dừng và bàn giao.
- **Lai**: cùng graph đó, nhưng bước hỏng thì quay lại lập kế hoạch cho các bước còn lại, tối đa 3 lần.

## Kết quả đánh giá

| Kịch bản | ReAct | Plan-then-Execute | Lai |
|---|---|---|---|
| `chuan`: chỉ VN122 hợp lệ | Đạt mục tiêu | Đạt mục tiêu | Đạt mục tiêu |
| `doi_gia`: VN122 tăng giá lúc kiểm ghế | Đạt mục tiêu | Kế hoạch hỏng | Đạt mục tiêu |
| `can_duyet`: chỉ còn vé không hoàn | Cần con người | Cần con người | Cần con người |
| `loi_dich_vu`: `check_seat` timeout | Phát hiện lặp | Kế hoạch hỏng | Phát hiện lặp |
| `ngan_sach`: chỉ 6 lần gọi model | Hết ngân sách | Kế hoạch hỏng | Hết ngân sách |

| Mẫu | Đạt mục tiêu (chuan + doi_gia) | Tổng gọi model | Tổng token ước lượng | Harness chặn |
|---|---:|---:|---:|---:|
| ReAct | 2/2 | 32 | 7.080 | 5 |
| Plan-then-Execute | 1/2 | 21 | 3.153 | 0 |
| Lai | 2/2 | 30 | 5.648 | 0 |

- **ReAct** linh hoạt nhưng tốn token nhất, và hay định làm sai nên bị harness chặn nhiều nhất.
- **Plan-then-Execute** rẻ nhất nhưng dừng ngay khi môi trường thay đổi.
- **Lai** đạt kết quả như ReAct với chi phí thấp hơn khoảng 20%.

Trong cả 15 lượt chạy không có vé sai nào lọt qua và không còn chỗ giữ treo.

## Hạn chế

- Kết quả trên chạy bằng model giả lập để tái hiện được. Tùy chọn `--that` đã có nhưng chưa chạy thử với model thật.
- Token là ước lượng (số ký tự chia 4), không phải số token từ API.
