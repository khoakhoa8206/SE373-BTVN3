# -*- coding: utf-8 -*-
"""Test BTVN#3. Chạy:  python -m unittest -v test_btvn3"""
import unittest

import btvn3_dat_ve as m


def _kich_ban(ten: str, ngan_sach: int = 12) -> m.Harness:
    m._kb = m.KICH_BAN[ten]
    m._dat_cho.clear()
    return m.Harness(m.RangBuoc(), ngan_sach=ngan_sach, in_trace=False)


def _goi(h: m.Harness, ten: str, **args) -> dict:
    return h.thuc_thi(ten, args, lambda: m._goi_tool(ten, args))


def _tim(h: m.Harness) -> dict:
    return _goi(h, "search_flights", origin="SGN", dest="DAD", date="2026-10-07")


class TestTool(unittest.TestCase):
    def setUp(self):
        _kich_ban("chuan")

    def test_ngay_sai_dinh_dang_tra_loi_co_goi_y(self):
        obs = m.search_flights("SGN", "DAD", "07/10")
        self.assertEqual(obs["status"], "invalid_param")
        self.assertIn("YYYY-MM-DD", obs["hint"])

    def test_khong_co_chuyen_la_list_rong_khong_phai_dict_rong(self):
        self.assertEqual(m.search_flights("SGN", "HUI", "2026-10-07"), {"status": "ok", "flights": []})

    def test_check_seat_tra_gia_moi(self):
        _kich_ban("doi_gia")
        self.assertEqual(m.check_seat("VN122")["price"], 2_050_000)

    def test_dich_vu_loi_tra_timeout(self):
        _kich_ban("loi_dich_vu")
        self.assertEqual(m.check_seat("VN122")["error"], "timeout")

    def test_sai_tham_so_khong_lam_sap(self):
        self.assertEqual(m._goi_tool("check_seat", {"chuyen": "VN122"})["status"], "invalid_param")


class TestRangBuoc(unittest.TestCase):
    rb = m.RangBuoc()

    def test_chuyen_hop_le(self):
        self.assertEqual(self.rb.vi_pham(m.CHUYEN["VN122"]), [])

    def test_bay_chieu_vi_pham_gio(self):
        self.assertIn("cất cánh 15:40", self.rb.vi_pham(m.CHUYEN["QH118"])[0])

    def test_vuot_tran_gia(self):
        self.assertIn("vượt trần", self.rb.vi_pham(m.CHUYEN["VN124"])[0])

    def test_cau_hoi_sinh_tu_du_lieu(self):
        self.assertEqual(self.rb.cau_hoi(), "Đặt giúp tôi vé SGN → DAD ngày 07/10/2026, "
                                            "cất cánh trước 12:00, giá dưới 2.000.000đ.")


class TestHarness(unittest.TestCase):
    def test_chan_dat_chuyen_chua_thay(self):
        h = _kich_ban("chuan")
        self.assertEqual(_goi(h, "book_seat", flight="VN122", seat="12A")["status"], "denied")
        self.assertEqual(m._dat_cho, {})

    def test_chan_dat_chuyen_vi_pham_rang_buoc(self):
        h = _kich_ban("chuan")
        _tim(h)
        obs = _goi(h, "book_seat", flight="QH118", seat="12A")
        self.assertEqual(obs["status"], "denied")
        self.assertEqual((h.bi_chan, m._dat_cho), (1, {}))
        self.assertIsNone(h.ket_qua)                     # chặn chứ không dừng: model còn sửa được

    def test_rang_buoc_dung_gia_moi_nhat_tu_check_seat(self):
        h = _kich_ban("doi_gia")
        _tim(h)
        _goi(h, "check_seat", flight="VN122")
        self.assertIn("vượt trần", _goi(h, "book_seat", flight="VN122", seat="12A")["vi_pham"][0])

    def test_ve_khong_hoan_phai_cho_duyet_truoc_khi_dat(self):
        h = _kich_ban("can_duyet")
        _tim(h)
        self.assertEqual(_goi(h, "book_seat", flight="VJ604", seat="12A")["status"], "stopped")
        self.assertEqual(h.ket_qua["kieu"], "Cần con người")
        self.assertIn("vé không hoàn", h.ket_qua["ban_giao"]["cau_hoi_cho_nguoi"])
        self.assertEqual(m._dat_cho, {})                 # tool chưa hề được gọi

    def test_pay_chi_nhan_ma_tu_book_seat(self):
        h = _kich_ban("chuan")
        self.assertEqual(_goi(h, "pay", code="BK999", method="corp_card")["status"], "denied")

    def test_tool_la_bi_chan(self):
        h = _kich_ban("chuan")
        self.assertEqual(_goi(h, "cancel_flight", code="BK001")["error"], "unknown_tool")

    def test_lap_cung_tool_args_thi_dung_co_ban_giao(self):
        h = _kich_ban("loi_dich_vu")
        _tim(h)
        _goi(h, "check_seat", flight="VN122")
        self.assertEqual(_goi(h, "check_seat", flight="VN122")["status"], "stopped")
        self.assertEqual(h.ket_qua["kieu"], "Phát hiện lặp")
        self.assertEqual(set(h.ket_qua["ban_giao"]),
                         {"stop_reason", "da_thu", "trang_thai", "cau_hoi_cho_nguoi"})

    def test_sau_khi_dung_khong_goi_tool_nua(self):
        h = _kich_ban("chuan")
        h._dung("Hết ngân sách", "x", "y")
        self.assertEqual(_tim(h)["status"], "stopped")
        self.assertEqual(h.goi_tool, 0)

    def test_hoan_thanh_kiem_bang_get_booking(self):
        h = _kich_ban("chuan")
        _tim(h)
        _goi(h, "check_seat", flight="VN122")
        code = _goi(h, "book_seat", flight="VN122", seat="12A")["code"]
        _goi(h, "pay", code=code, method="corp_card")
        self.assertIsNone(h.xet_dung())                  # đã trả tiền nhưng chưa kiểm chứng
        _goi(h, "get_booking", code=code)
        self.assertEqual(h.xet_dung()["kieu"], "Đạt mục tiêu")

    def test_model_tu_tuyen_bo_xong_khong_duoc_tin(self):
        h = _kich_ban("chuan")
        self.assertEqual(h.ket_thuc_model("Đã đặt VN999 giá 1.200.000đ.")["kieu"], "Chưa đạt mục tiêu")

    def test_ban_giao_ghi_tac_dung_phu(self):
        h = _kich_ban("chuan")
        _tim(h)
        _goi(h, "check_seat", flight="VN122")
        _goi(h, "book_seat", flight="VN122", seat="12A")
        h._dung("Hết ngân sách", "x", "y")
        self.assertEqual(h.ket_qua["ban_giao"]["trang_thai"]["tac_dung_phu"], ["BK001 VN122 đang giữ chỗ"])

    def test_het_ngan_sach(self):
        h = _kich_ban("chuan", ngan_sach=2)
        h.goi_model = 2
        self.assertEqual(h.xet_dung()["kieu"], "Hết ngân sách")


class TestKeHoach(unittest.TestCase):
    def test_ke_hoach_hop_le(self):
        self.assertEqual(len(m._doc_ke_hoach('[{"tool":"pay"},{"tool":"get_booking"}]')), 2)

    def test_chap_nhan_code_fence(self):
        self.assertIsNotNone(m._doc_ke_hoach('```json\n[{"tool":"get_booking"}]\n```'))

    def test_tu_choi_ke_hoach_khong_kiem_chung(self):
        self.assertIsNone(m._doc_ke_hoach('[{"tool":"pay"}]'))

    def test_tu_choi_tool_la_va_rac(self):
        self.assertIsNone(m._doc_ke_hoach('[{"tool":"cancel_flight"},{"tool":"get_booking"}]'))
        self.assertIsNone(m._doc_ke_hoach("không phải JSON"))
        self.assertIsNone(m._doc_ke_hoach("[]"))


class TestTichHop(unittest.TestCase):
    """Chạy agent thật (create_agent / StateGraph) với model giả lập."""

    MONG_DOI = {
        ("chuan", "react"): "Đạt mục tiêu", ("chuan", "plan"): "Đạt mục tiêu", ("chuan", "lai"): "Đạt mục tiêu",
        ("doi_gia", "react"): "Đạt mục tiêu", ("doi_gia", "plan"): "Kế hoạch hỏng",
        ("doi_gia", "lai"): "Đạt mục tiêu",
        ("can_duyet", "react"): "Cần con người", ("can_duyet", "plan"): "Cần con người",
        ("can_duyet", "lai"): "Cần con người",
        ("loi_dich_vu", "react"): "Phát hiện lặp", ("loi_dich_vu", "plan"): "Kế hoạch hỏng",
        ("loi_dich_vu", "lai"): "Phát hiện lặp",
        ("ngan_sach", "react"): "Hết ngân sách", ("ngan_sach", "plan"): "Kế hoạch hỏng",
        ("ngan_sach", "lai"): "Hết ngân sách",
    }

    def test_ket_cuc_tung_kich_ban_va_mau(self):
        for (kb, mau), kieu in self.MONG_DOI.items():
            with self.subTest(kich_ban=kb, mau=mau):
                r = m.chay(mau, kb, in_trace=False)
                self.assertEqual(r["kieu"], kieu)
                self.assertEqual(r["ve_sai"], 0)
                self.assertEqual(r["giu_cho_treo"], 0)
                self.assertLessEqual(r["goi_model"], m.KICH_BAN[kb].get("ngan_sach", 12))

    def test_dat_dung_chuyen(self):
        m.chay("react", "doi_gia", in_trace=False)
        self.assertEqual([b["flight"] for b in m._dat_cho.values() if b["paid"]], ["VN126"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
