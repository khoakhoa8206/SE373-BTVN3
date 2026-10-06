# -*- coding: utf-8 -*-
"""SE373 · Buổi 03 · Hai lớp harness viết tay.

Đây là phần framework KHÔNG làm hộ. Cả hai đều là Python thường, không gọi
model, không phụ thuộc LangChain, nên cắm được vào bất kỳ vòng lặp nào.

    LoopDetector   phát hiện lặp và bế tắc, chạy sau mỗi observation   (S41, S48, S49)
    kiem_can_cu    đối chiếu câu trả lời với kết quả tool đã nhận      (S61, S62)

Chạy thử:  python3 lib/harness.py
"""
from collections import deque
import re


# ===================================================== 1 · PHÁT HIỆN LẶP
class LoopDetector:
    """Ba tín hiệu, theo đúng slide S48.

    1. trùng action      cùng (tool, args) lặp lại trong cửa sổ vài vòng
    2. trùng observation tham số khác nhau nhưng kết quả trả về giống hệt
    3. không tiến triển  một đại lượng của bài toán đứng yên qua N vòng

    Tín hiệu 3 bắt được kiểu bế tắc mà hai tín hiệu đầu bỏ sót: agent đổi
    tool mỗi vòng nhưng vẫn đứng yên.
    """

    def __init__(self, window=6, repeat_k=2, same_obs_k=4, stall_n=5):
        self.recent = deque(maxlen=window)   # dấu vân tay (tool, args)
        self.obs = deque(maxlen=window)      # dấu vân tay observation
        self.k, self.k_obs, self.n = repeat_k, same_obs_k, stall_n
        self.last_progress, self.stall = None, 0

    def check(self, tool: str, args: dict, observation=None, progress=None):
        """Trả chuỗi mô tả nếu phát hiện lặp hoặc bế tắc, None nếu bình thường."""
        fp = (tool, repr(sorted(args.items())))
        if self.recent.count(fp) + 1 >= self.k:
            return (f"LOOP · '{tool}' gọi {self.recent.count(fp) + 1} lần "
                    f"với cùng tham số trong {self.recent.maxlen} vòng gần nhất")
        self.recent.append(fp)

        if observation is not None:
            ofp = repr(observation)
            if self.obs.count(ofp) + 1 >= self.k_obs:
                return (f"LOOP · {self.obs.count(ofp) + 1} lời gọi khác tham số "
                        f"nhưng trả về cùng một kết quả")
            self.obs.append(ofp)

        if progress is not None:
            self.stall = self.stall + 1 if progress == self.last_progress else 0
            self.last_progress = progress
            if self.stall >= self.n:
                return f"STALL · tiến triển đứng yên ở {progress!r} qua {self.stall} vòng"
        return None


def ban_giao(ly_do: str, da_thu: list, trang_thai: dict, cau_hoi: str) -> dict:
    """Dừng bất thường thì bàn giao đủ ba thứ, không break im lặng (S50)."""
    return {"stop_reason": ly_do,
            "da_thu": da_thu,
            "trang_thai": trang_thai,
            "cau_hoi_cho_nguoi": cau_hoi}


# ================================================== 2 · KIỂM TRA CĂN CỨ
# số tiền, giờ, ngày, mã đơn, bốn số cuối thẻ, nhiệt độ
MAU_DU_KIEN = [
    r"\d{1,3}(?:\.\d{3})+(?:đ|\s?VNĐ|\s?đồng)?",   # 1.250.000đ
    r"\d{1,2}:\d{2}",                              # 18:40
    r"\d{1,2}/\d{1,2}(?:/\d{2,4})?",               # 06/09
    r"[A-Z]{2,5}-\d{3,8}",                         # ORD-88123
    r"\b\d{4}\b",                                  # 4412
    r"\d{1,2}°C",                                  # 31°C
]


def trich_du_kien(text: str) -> list:
    """Lấy mọi số, ngày, giờ và mã định danh xuất hiện trong câu trả lời."""
    ra = []
    for mau in MAU_DU_KIEN:
        ra += re.findall(mau, text)
    return list(dict.fromkeys(ra))      # giữ thứ tự, bỏ trùng


def kiem_can_cu(cau_tra_loi: str, ket_qua_tool: list) -> dict:
    """ket_qua_tool: danh sách nội dung các ToolMessage đã nhận trong phiên."""
    nguon = " ".join(str(x) for x in ket_qua_tool)
    chi_tiet = [{"du_kien": d, "co_nguon": d in nguon} for d in trich_du_kien(cau_tra_loi)]
    thieu = [c["du_kien"] for c in chi_tiet if not c["co_nguon"]]
    return {"dat": not thieu, "chi_tiet": chi_tiet, "khong_co_nguon": thieu}


def chan_truoc_khi_tra_loi(cau_tra_loi: str, ket_qua_tool: list) -> str:
    """Trả câu an toàn. Không tự sửa số liệu, chỉ chặn và nói rõ phần thiếu nguồn."""
    kq = kiem_can_cu(cau_tra_loi, ket_qua_tool)
    if kq["dat"]:
        return cau_tra_loi
    return ("Chưa trả lời được đầy đủ. Các dữ kiện sau không truy được về kết quả tool nào: "
            + " · ".join(kq["khong_co_nguon"])
            + ". Cần bổ sung tool cung cấp thông tin này, hoặc chuyển cho nhân viên hỗ trợ.")


if __name__ == "__main__":
    print("── LoopDetector · tái hiện Trace A ──")
    det = LoopDetector()
    for vong, ten in enumerate(
            ["Vung Tau", "Vũng Tàu", "Vung Tau City", "Vung Tau"], 1):
        canh_bao = det.check("weather_forecast", {"town": ten},
                             observation={"error": "not found"}, progress=0)
        print(f"  V{vong} weather_forecast(town={ten!r}) → {canh_bao or 'bình thường'}")
        if canh_bao:
            break

    print("\n── kiem_can_cu · tái hiện Trace B ──")
    ket_qua_tool = [
        '{"order_id":"ORD-88123","status":"in_transit","carrier":"GHN"}',
        '{"last_scan":"Kho trung chuyển Bình Dương","scanned_at":"06/09 18:40"}',
    ]
    cau = ("Đơn ORD-88123 đang trên đường giao, quét gần nhất lúc 18:40 ngày 06/09. "
           "Tổng tiền 1.250.000đ, đã gồm 30.000đ phí vận chuyển, thẻ VISA đuôi 4412.")
    for c in kiem_can_cu(cau, ket_qua_tool)["chi_tiet"]:
        print(("   có nguồn " if c["co_nguon"] else "   BỊA     "), c["du_kien"])
    print("\n  Câu được phép trả ra:\n  " + chan_truoc_khi_tra_loi(cau, ket_qua_tool))
