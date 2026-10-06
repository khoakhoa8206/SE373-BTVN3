# -*- coding: utf-8 -*-
"""BTVN#3 · Agent đặt vé máy bay bằng LangChain / LangGraph.

Ba mẫu thiết kế dùng CHUNG một bộ tool mockup và CHUNG một lớp harness,
nên khác biệt kết quả chỉ đến từ mẫu thiết kế:

    react   create_agent + HarnessMiddleware · model chọn từng bước
    plan    StateGraph · lập kế hoạch một lần rồi thực thi tuần tự, bước hỏng là dừng
    lai     StateGraph · như plan, nhưng bước hỏng thì lập lại kế hoạch (tối đa 3 lần)

Các lớp harness (code của mình, không phải của model):
    RangBuoc        ràng buộc là dữ liệu, kiểm trước book_seat
    hoan_thanh()    tiêu chí hoàn thành kiểm bằng code trên kết quả get_booking
    _can_duyet()    kiểm quyền trước khi thực thi book_seat
    LoopDetector    phát hiện lặp + bế tắc (lib/harness.py của demo)
    ngân sách       số lần gọi model, kiểm CUỐI cùng (slide 35)
    ban_giao()      dừng bất thường luôn trả gói bàn giao

Chạy:
    python btvn3_dat_ve.py --mau react --kich-ban chuan    # trace một lượt chạy
    python btvn3_dat_ve.py --danh-gia                      # 5 kịch bản × 3 mẫu
    python btvn3_dat_ve.py --mau lai --that                # model thật, đọc SE373_MODEL
"""
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from lib.harness import LoopDetector, ban_giao, chan_truoc_khi_tra_loi


def _vnd(x: int) -> str:
    return f"{x:,}".replace(",", ".") + "đ"


# ================================================================ TOOL MOCKUP
# Dữ liệu tĩnh SGN → DAD ngày 2026-10-07. Mỗi kịch bản bật một tập chuyến.
CHUYEN = {
    "QH118": dict(depart="15:40", price=1_640_000, refundable=True),
    "VJ604": dict(depart="06:05", price=1_390_000, refundable=False),
    "VN122": dict(depart="08:10", price=1_850_000, refundable=True),
    "VN126": dict(depart="11:20", price=1_880_000, refundable=True),
    "VN124": dict(depart="10:30", price=2_150_000, refundable=True),
}

KICH_BAN = {
    # có đúng một chuyến thoả mọi ràng buộc: VN122
    "chuan":       dict(chuyen=["QH118", "VN122", "VN124"]),
    # môi trường đổi: lúc check_seat, VN122 tăng giá vượt trần → phải chuyển sang VN126
    "doi_gia":     dict(chuyen=["QH118", "VN122", "VN126", "VN124"], gia_moi={"VN122": 2_050_000}),
    # chuyến hợp lệ duy nhất là vé không hoàn → phải chờ người duyệt
    "can_duyet":   dict(chuyen=["QH118", "VJ604", "VN124"]),
    # dịch vụ check_seat sập → mọi lần gọi đều timeout
    "loi_dich_vu": dict(chuyen=["QH118", "VN122", "VN124"], timeout=True),
    # như doi_gia nhưng ngân sách chỉ 6 lần gọi model
    "ngan_sach":   dict(chuyen=["QH118", "VN122", "VN126", "VN124"], gia_moi={"VN122": 2_050_000},
                        ngan_sach=6),
}

# ponytail: trạng thái toàn cục cho mockup, mỗi lượt chạy reset; cần chạy song song thì gói vào class
_kb: dict = KICH_BAN["chuan"]
_dat_cho: dict = {}


def _gia(flight: str) -> int:
    return _kb.get("gia_moi", {}).get(flight, CHUYEN[flight]["price"])


def search_flights(origin: str, dest: str, date: str) -> dict:
    """Tìm chuyến bay một chiều. date bắt buộc dạng YYYY-MM-DD."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return {"status": "invalid_param", "param": "date", "hint": "Dùng YYYY-MM-DD, ví dụ 2026-10-07"}
    if (origin.upper(), dest.upper(), date) != ("SGN", "DAD", "2026-10-07"):
        return {"status": "ok", "flights": []}            # thật sự không có chuyến
    return {"status": "ok", "flights": [{"flight": f, **CHUYEN[f]} for f in _kb["chuyen"]]}


def check_seat(flight: str) -> dict:
    """Kiểm tra ghế trống và giá HIỆN TẠI của một chuyến (giá có thể khác lúc tìm)."""
    if _kb.get("timeout"):
        return {"status": "error", "error": "timeout", "hint": "Dịch vụ kiểm tra ghế đang lỗi, thử lại sau"}
    if flight not in _kb["chuyen"]:
        return {"status": "error", "error": "flight_not_found", "hint": "Lấy mã chuyến từ search_flights"}
    return {"status": "ok", "flight": flight, "price": _gia(flight), "seats": ["12A", "12B", "14C"]}


def book_seat(flight: str, seat: str) -> dict:
    """Giữ chỗ một ghế. Trả mã đặt chỗ ở trạng thái held, chưa thanh toán."""
    if flight not in _kb["chuyen"]:
        return {"status": "error", "error": "flight_not_found", "hint": "Lấy mã chuyến từ search_flights"}
    code = f"BK{len(_dat_cho) + 1:03d}"
    _dat_cho[code] = {"code": code, "flight": flight, "seat": seat, "date": "2026-10-07",
                      **CHUYEN[flight], "price": _gia(flight), "state": "held", "paid": False}
    return {"status": "ok", **_dat_cho[code]}


def pay(code: str, method: str) -> dict:
    """Thanh toán một mã đặt chỗ đang held."""
    if code not in _dat_cho:
        return {"status": "error", "error": "booking_not_found", "hint": "Dùng mã trả về từ book_seat"}
    _dat_cho[code].update(state="confirmed", paid=True)
    return {"status": "ok", "code": code, "paid": True}


def get_booking(code: str) -> dict:
    """Đọc lại trạng thái đặt chỗ từ hệ thống (dùng để kiểm chứng)."""
    if code not in _dat_cho:
        return {"status": "error", "error": "booking_not_found", "hint": "Dùng mã trả về từ book_seat"}
    return {"status": "ok", **_dat_cho[code]}


TOOLS = {f.__name__: f for f in (search_flights, check_seat, book_seat, pay, get_booking)}
TOOLS_LC = [tool(f) for f in TOOLS.values()]
CHANG = list(TOOLS)          # thứ tự quy trình đặt vé, dùng làm đại lượng tiến triển


def _goi_tool(ten: str, args: dict) -> dict:
    try:
        return TOOLS[ten](**args)
    except TypeError as e:                                # model truyền sai tham số
        return {"status": "invalid_param", "error": str(e)}


def _doc(x: Any) -> Any:
    try:
        return json.loads(x)
    except Exception:
        return x


# ==================================================================== HARNESS
@dataclass(frozen=True)
class RangBuoc:
    """Ràng buộc là dữ liệu: giữ yêu cầu ở một chỗ cố định, kiểm lại trước khi đặt (slide 63)."""
    origin: str = "SGN"
    dest: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"
    max_price: int = 2_000_000

    def vi_pham(self, chuyen: dict) -> list:
        loi = []
        if chuyen.get("date", self.date) != self.date:
            loi.append(f"ngày {chuyen['date']}, cần {self.date}")
        if chuyen["depart"] >= self.depart_before:
            loi.append(f"cất cánh {chuyen['depart']}, cần trước {self.depart_before}")
        if chuyen["price"] > self.max_price:
            loi.append(f"giá {_vnd(chuyen['price'])} vượt trần {_vnd(self.max_price)}")
        return loi

    def cau_hoi(self) -> str:
        y, m, d = self.date.split("-")
        return (f"Đặt giúp tôi vé {self.origin} → {self.dest} ngày {d}/{m}/{y}, "
                f"cất cánh trước {self.depart_before}, giá dưới {_vnd(self.max_price)}.")


HAN_MUC_TU_DUYET = 1_900_000      # trên mức này, hoặc vé không hoàn, phải có người duyệt


class Harness:
    """Một instance cho một lượt chạy. Ba mẫu thiết kế đều đi qua đúng các hàm này."""

    def __init__(self, rb: RangBuoc, ngan_sach: int = 12, in_trace: bool = True):
        self.rb, self.ngan_sach, self.in_trace = rb, ngan_sach, in_trace
        self.det = LoopDetector(window=6, repeat_k=2, stall_n=5)
        self.nhat_ky = []        # (tool, args, observation) theo thứ tự
        self.chuyen = {}         # flight -> thông tin mới nhất (search, rồi check_seat ghi đè giá)
        self.goi_model = self.goi_tool = self.bi_chan = self.token = 0
        self.ket_qua = None      # dict khi harness đã quyết định dừng

    def log(self, s: str) -> None:
        if self.in_trace:
            print(s)

    # ------------------------------------------------- trước mỗi lần gọi model
    def dem_model(self, messages: list) -> None:
        self.goi_model += 1
        # ponytail: token ước lượng = ký tự/4, đủ để so sánh tương đối giữa các mẫu
        self.token += sum(len(str(m.content)) + len(str(getattr(m, "tool_calls", "") or ""))
                          for m in messages) // 4

    # --------------------------------------------------- mỗi lần gọi tool
    def thuc_thi(self, ten: str, args: dict, goi) -> dict:
        if self.ket_qua:
            return {"status": "stopped", "reason": self.ket_qua["kieu"]}
        if ten not in TOOLS:
            return self._chan(ten, args, {"status": "error", "error": "unknown_tool", "valid": CHANG})

        canh_bao = self.det.check(ten, args, progress=self.tien_trien())
        if canh_bao:
            self.log(f"[{len(self.nhat_ky) + 1:>2}] {ten}({_args(args)})  ✋ {canh_bao}")
            kieu = "Phát hiện lặp" if canh_bao.startswith("LOOP") else "Bế tắc"
            self._dung(kieu, canh_bao,
                       f"{ten} không cho kết quả mới. Thử lại sau, hay chuyển nhân viên đặt thủ công?")
            return {"status": "stopped", "reason": canh_bao}

        if ten == "book_seat":
            tu_choi = self._kiem_rang_buoc(args.get("flight"))
            if tu_choi:
                return self._chan(ten, args, tu_choi)
            ly_do = self._can_duyet(args["flight"])
            if ly_do:
                c = self.chuyen[args["flight"]]
                self.log(f"[{len(self.nhat_ky) + 1:>2}] {ten}({_args(args)})  ✋ cần duyệt: {ly_do}")
                self._dung("Cần con người", f"APPROVAL · book_seat {args['flight']}: {ly_do}",
                           f"Duyệt đặt {args['flight']} {c['depart']} giá {_vnd(c['price'])} "
                           f"({ly_do}) không?")
                return {"status": "stopped", "reason": "cần duyệt"}
        if ten == "pay" and args.get("code") not in self._ma_da_giu():
            return self._chan(ten, args, {"status": "denied", "reason": "mã không đến từ book_seat"})

        obs = goi()
        self.goi_tool += 1
        self._ghi(ten, args, obs)
        return obs

    def _kiem_rang_buoc(self, flight):
        c = self.chuyen.get(flight)
        if not c:
            return {"status": "denied", "reason": "chưa thấy chuyến này trong search_flights/check_seat"}
        loi = self.rb.vi_pham(c)
        if loi:
            return {"status": "denied", "flight": flight, "vi_pham": loi,
                    "hint": "Chọn chuyến khác thoả mọi ràng buộc"}
        return None

    def _can_duyet(self, flight) -> str:
        c, ly_do = self.chuyen[flight], []
        if not c["refundable"]:
            ly_do.append("vé không hoàn")
        if c["price"] > HAN_MUC_TU_DUYET:
            ly_do.append(f"vượt hạn mức tự duyệt {_vnd(HAN_MUC_TU_DUYET)}")
        return ", ".join(ly_do)

    def _ma_da_giu(self) -> set:
        return {o["code"] for t, _, o in self.nhat_ky if t == "book_seat" and o.get("status") == "ok"}

    def _chan(self, ten, args, obs) -> dict:
        self.bi_chan += 1
        self._ghi(ten, args, obs, dau="✗ CHẶN ")
        return obs

    def _ghi(self, ten, args, obs, dau="") -> None:
        self.nhat_ky.append((ten, args, obs))
        ok = isinstance(obs, dict) and obs.get("status") == "ok"
        if ten == "search_flights" and ok:
            self.chuyen = {f["flight"]: dict(f) for f in obs["flights"]}
        elif ten == "check_seat" and ok and obs["flight"] in self.chuyen:
            self.chuyen[obs["flight"]]["price"] = obs["price"]
        hien = json.dumps(obs, ensure_ascii=False)
        self.log(f"[{len(self.nhat_ky):>2}] {ten}({_args(args)}) → {dau}{hien[:110]}")

    def tien_trien(self) -> int:
        """Chặng xa nhất đã thành công trong quy trình đặt vé (0..5)."""
        return max([CHANG.index(t) + 1 for t, _, o in self.nhat_ky
                    if isinstance(o, dict) and o.get("status") == "ok"], default=0)

    # --------------------------------------------- tiêu chí hoàn thành bằng code
    def hoan_thanh(self):
        for t, _, o in reversed(self.nhat_ky):
            if t == "get_booking" and o.get("status") == "ok":
                if o["state"] == "confirmed" and o["paid"] and not self.rb.vi_pham(o):
                    return o
        return None

    # ------------------------------------------- xét dừng sau mỗi observation
    def xet_dung(self):
        """Thứ tự slide 35: hoàn thành → (lặp/bế tắc/quyền đã xét lúc gọi tool) → ngân sách cuối."""
        if self.ket_qua:
            return self.ket_qua
        ve = self.hoan_thanh()
        if ve:
            self.ket_qua = {"kieu": "Đạt mục tiêu", "binh_thuong": True, "noi_dung":
                            f"Đã đặt {ve['flight']} ngày {ve['date']} {ve['depart']}, ghế {ve['seat']}, "
                            f"mã {ve['code']}, {_vnd(ve['price'])} · đã thanh toán, get_booking xác nhận."}
        elif self.goi_model >= self.ngan_sach:
            self._dung("Hết ngân sách", f"BUDGET · đã dùng {self.goi_model}/{self.ngan_sach} lần gọi model",
                       "Nới ràng buộc (giờ bay, giá) hay tăng ngân sách?")
        return self.ket_qua

    def ket_thuc_model(self, cau: str) -> dict:
        """Model tự cho là xong (thôi gọi tool). Không tin: kiểm bằng code."""
        if self.hoan_thanh():
            return self.xet_dung()
        an_toan = chan_truoc_khi_tra_loi(cau, [json.dumps(o, ensure_ascii=False) for *_, o in self.nhat_ky])
        self._dung("Chưa đạt mục tiêu", f"DONE? · model dừng nhưng tiêu chí hoàn thành chưa thoả: {an_toan}",
                   "Nới ràng buộc nào (giờ bay, giá), hay dừng tìm?")
        return self.ket_qua

    def _dung(self, kieu, ly_do, cau_hoi) -> None:
        if self.ket_qua:
            return
        da_tra = {o["code"] for t, _, o in self.nhat_ky if t == "pay" and o.get("status") == "ok"}
        tac_dung_phu = [f"{o['code']} {o['flight']} " + ("đã trả tiền" if o["code"] in da_tra else "đang giữ chỗ")
                        for t, _, o in self.nhat_ky if t == "book_seat" and o.get("status") == "ok"]
        bg = ban_giao(
            ly_do=ly_do,
            da_thu=[f"{t}({_args(a)})→{o.get('status')}" for t, a, o in self.nhat_ky],
            trang_thai={"tien_trien": f"{self.tien_trien()}/5 chặng", "goi_model": self.goi_model,
                        "goi_tool": self.goi_tool, "tac_dung_phu": tac_dung_phu or "không"},
            cau_hoi=cau_hoi,
        )
        self.ket_qua = {"kieu": kieu, "binh_thuong": kieu == "Cần con người",
                        "noi_dung": _in_ban_giao(bg), "ban_giao": bg}


def _args(a: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in a.items())


def _in_ban_giao(b: dict) -> str:
    return ("DỪNG · " + b["stop_reason"] + "\n"
            "  Đã thử         : " + " | ".join(b["da_thu"]) + "\n"
            "  Trạng thái     : " + str(b["trang_thai"]) + "\n"
            "  Hỏi người dùng : " + b["cau_hoi_cho_nguoi"])


# ======================================================= MỘT · ReAct
PROMPT_REACT = ("Bạn là agent đặt vé máy bay. Chỉ dùng tool để lấy thông tin; mã chuyến, giá, "
                "mã đặt chỗ phải lấy từ kết quả tool. Đặt xong thì kiểm lại bằng get_booking.")


class HarnessMiddleware(AgentMiddleware):
    """Cắm Harness vào vòng lặp của create_agent."""

    def __init__(self, h: Harness):
        super().__init__()
        self.h = h

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):            # = sau mỗi observation
        kq = self.h.xet_dung()
        if kq:
            return {"jump_to": "end", "messages": [AIMessage(content=kq["noi_dung"])]}
        self.h.dem_model(state["messages"])
        return None

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime):
        cuoi = state["messages"][-1]
        if cuoi.tool_calls:
            return None
        kq = self.h.ket_thuc_model(str(cuoi.content))
        return {"jump_to": "end", "messages": [AIMessage(content=kq["noi_dung"])]}

    def wrap_tool_call(self, request, handler):
        c = request.tool_call
        obs = self.h.thuc_thi(c["name"], c["args"], lambda: _doc(handler(request).content))
        return ToolMessage(content=json.dumps(obs, ensure_ascii=False), tool_call_id=c["id"], name=c["name"])


# ============================================ HAI + BA · Plan-then-Execute và Lai
PROMPT_PLANNER = ("[PLANNER] Lập kế hoạch đặt vé. Trả về DUY NHẤT một JSON list, mỗi phần tử "
                  '{"tool": <tên tool>, "mo_ta": <việc cần làm>}. Tool hợp lệ: ' + ", ".join(CHANG) +
                  ". Bước cuối phải là get_booking để kiểm chứng. Nếu đã có lịch sử, chỉ lập các bước còn lại.")
PROMPT_EXECUTOR = ("[EXECUTOR] Thực hiện đúng một bước của kế hoạch bằng đúng một lời gọi tool {tool}: "
                   "{mo_ta}. Tuân thủ mọi ràng buộc trong yêu cầu của người dùng.")


class TrangThaiKH(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    ke_hoach: list
    buoc: int
    so_lan_lap: int
    hong: str


def _doc_ke_hoach(text: str):
    """Tiêu chí 'schema hợp lệ' cho kế hoạch: sai thì không chạy (slide 44)."""
    try:
        kh = json.loads(re.sub(r"^```(json)?|```$", "", str(text).strip()).strip())
        assert isinstance(kh, list) and kh and kh[-1]["tool"] == "get_booking"
        assert all(b["tool"] in TOOLS for b in kh)
        return kh
    except Exception:
        return None


def tao_plan_graph(model, h: Harness, cho_lap_lai: bool):
    model_tool = model.bind_tools(TOOLS_LC)

    def lap_ke_hoach(s: TrangThaiKH):
        msgs = [SystemMessage(PROMPT_PLANNER), *s["messages"]]
        if s.get("hong"):
            msgs.append(HumanMessage(f"Bước vừa hỏng: {s['hong']}. Lập lại các bước còn lại."))
        h.dem_model(msgs)
        kh = _doc_ke_hoach(model.invoke(msgs).content)
        if kh is None:
            h._dung("Kế hoạch không hợp lệ", "PLAN · planner không trả được kế hoạch hợp lệ",
                    "Nới ràng buộc nào (giờ bay, giá), hay dừng tìm?")
        else:
            h.log(("KẾ HOẠCH LẠI: " if s.get("hong") else "KẾ HOẠCH: ") + " → ".join(b["tool"] for b in kh))
        return {"ke_hoach": kh or [], "buoc": 0, "hong": "",
                "so_lan_lap": s.get("so_lan_lap", 0) + (1 if s.get("hong") else 0)}

    def thuc_thi(s: TrangThaiKH):
        buoc = s["ke_hoach"][s["buoc"]]
        msgs = [SystemMessage(PROMPT_EXECUTOR.format(**buoc)), *s["messages"]]
        h.dem_model(msgs)
        ai = model_tool.invoke(msgs)
        c = (ai.tool_calls or [None])[0]
        if not c or c["name"] != buoc["tool"]:
            de_xuat = c["name"] if c else "không gọi tool"
            h.log(f"     ⚠ bước {s['buoc'] + 1} cần {buoc['tool']}, executor đề xuất {de_xuat} → lệch kế hoạch")
            return {"hong": f"bước {s['buoc'] + 1} cần {buoc['tool']} nhưng không còn làm được ({de_xuat})"}
        obs = h.thuc_thi(c["name"], c["args"], lambda: _goi_tool(c["name"], c["args"]))
        tm = ToolMessage(json.dumps(obs, ensure_ascii=False), tool_call_id=c["id"], name=c["name"])
        return {"messages": [AIMessage(content=ai.content, tool_calls=[c]), tm], "buoc": s["buoc"] + 1,
                "hong": "" if obs.get("status") == "ok" else f"{c['name']} → {json.dumps(obs, ensure_ascii=False)}"}

    def dieu_huong(s: TrangThaiKH):
        if h.xet_dung():
            return END
        if s.get("hong"):
            if cho_lap_lai and s.get("so_lan_lap", 0) < 3:
                return "lap_ke_hoach"
            h._dung("Kế hoạch hỏng", f"PLAN · {s['hong']}", "Lập kế hoạch mới (chuyến khác), hay nới ràng buộc?")
            return END
        if s["buoc"] >= len(s["ke_hoach"]):
            h.ket_thuc_model("Đã chạy hết kế hoạch.")
            return END
        return "thuc_thi"

    g = StateGraph(TrangThaiKH)
    g.add_node("lap_ke_hoach", lap_ke_hoach)
    g.add_node("thuc_thi", thuc_thi)
    g.add_edge(START, "lap_ke_hoach")
    g.add_conditional_edges("lap_ke_hoach", dieu_huong, ["thuc_thi", END])
    g.add_conditional_edges("thuc_thi", dieu_huong, ["thuc_thi", "lap_ke_hoach", END])
    return g.compile()


# ================================================================ MODEL GIẢ LẬP
class ModelGia(BaseChatModel):
    """Không phải LLM: đọc câu hỏi + observation rồi quyết định theo luật, để chạy không cần API key.

    Ở vai ReAct nó cố tình mắc hai thói quen hay gặp ở LLM thật:
      · chép nguyên ngày "07/10" của người dùng vào search_flights (sai định dạng)
      · chọn chuyến rẻ nhất mà quên ràng buộc giờ bay / giá  → harness phải chặn
    Ở vai PLANNER / EXECUTOR (đọc kỹ yêu cầu khi lập kế hoạch) nó lọc theo ràng buộc.
    """

    @property
    def _llm_type(self) -> str:
        return "btvn3-model-gia"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ModelGia":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        he_thong = next((str(m.content) for m in messages if m.type == "system"), "")
        ai = _buoc_gia(messages, can_than=he_thong.startswith(("[PLANNER]", "[EXECUTOR]")))
        if he_thong.startswith("[PLANNER]"):          # kế hoạch = các chặng còn lại kể từ bước kế
            con = CHANG[CHANG.index(ai.tool_calls[0]["name"]):] if ai.tool_calls else []
            ai = AIMessage(content=json.dumps([{"tool": t, "mo_ta": MO_TA[t]} for t in con], ensure_ascii=False))
        return ChatResult(generations=[ChatGeneration(message=ai)])


MO_TA = {"search_flights": "tìm chuyến theo tuyến và ngày (YYYY-MM-DD)",
         "check_seat": "kiểm ghế + giá hiện tại của chuyến rẻ nhất thoả giờ bay và giá",
         "book_seat": "giữ ghế trên chuyến vừa kiểm, nếu vẫn thoả ràng buộc",
         "pay": "thanh toán mã đặt chỗ vừa giữ",
         "get_booking": "đọc lại đặt chỗ để kiểm chứng"}


def _buoc_gia(messages, can_than: bool) -> AIMessage:
    q = next(str(m.content) for m in messages if m.type == "human")
    o, d, dd, mm, yyyy, truoc, tran = re.search(
        r"(\w{3}) → (\w{3}) ngày (\d{2})/(\d{2})/(\d{4}).*?trước (\d{2}:\d{2}).*?dưới ([\d.]+)đ", q).groups()
    tran = int(tran.replace(".", ""))

    calls = {c["id"]: c for m in messages if m.type == "ai" for c in (m.tool_calls or [])}
    chuyen, loai, da_kiem = {}, set(), {}
    da_tim, giu, tra, xac_nhan, cuoi = False, None, None, None, None
    for m in messages:
        if m.type != "tool" or m.tool_call_id not in calls:
            continue
        c, obs = calls[m.tool_call_id], _doc(m.content)
        ten, a, ok = c["name"], c["args"], obs.get("status") == "ok"
        cuoi = (ten, a, obs)
        if ten == "search_flights" and ok:
            da_tim, chuyen = True, {f["flight"]: dict(f) for f in obs["flights"]}
        elif ten == "check_seat" and ok:
            da_kiem[a["flight"]] = obs
            chuyen[a["flight"]]["price"] = obs["price"]
        elif ten == "book_seat":
            if ok:
                giu = obs
            else:
                loai.add(a["flight"])
        elif ten == "pay" and ok:
            tra = obs
        elif ten == "get_booking" and ok and obs["state"] == "confirmed":
            xac_nhan = obs

    def goi(ten, **args):
        return AIMessage(content="", tool_calls=[
            {"name": ten, "args": args, "id": f"call_{len(calls) + 1}", "type": "tool_call"}])

    def hop_le(f):
        return not can_than or (f["depart"] < truoc and f["price"] <= tran)

    if xac_nhan:
        return AIMessage(content=f"Đã đặt xong {xac_nhan['flight']}, mã {xac_nhan['code']}.")
    if tra:
        return goi("get_booking", code=tra["code"])
    if giu:
        return goi("pay", code=giu["code"], method="corp_card")
    if cuoi and cuoi[0] == "check_seat" and cuoi[2].get("status") == "error":
        return goi("check_seat", **cuoi[1])                  # lỗi thì thử lại y hệt
    if not da_tim:
        sai_dinh_dang = cuoi is None and not can_than
        return goi("search_flights", origin=o, dest=d, date=f"{dd}/{mm}" if sai_dinh_dang else f"{yyyy}-{mm}-{dd}")
    for ma, kq in da_kiem.items():                           # đã kiểm ghế, chưa đặt
        if ma not in loai and kq["seats"] and hop_le(chuyen[ma]):
            return goi("book_seat", flight=ma, seat=kq["seats"][0])
    con = sorted((f for ma, f in chuyen.items() if ma not in loai and ma not in da_kiem and hop_le(f)),
                 key=lambda f: f["price"])
    if con:
        return goi("check_seat", flight=con[0]["flight"])
    return AIMessage(content="Không tìm được chuyến phù hợp với yêu cầu.")


def _model_that():
    from langchain.chat_models import init_chat_model
    ten = os.environ.get("SE373_MODEL")
    if not ten:
        raise SystemExit('Chưa đặt SE373_MODEL, ví dụ "anthropic:claude-sonnet-5-5" hoặc "openai:gpt-4.1-mini".')
    return init_chat_model(ten)


# ==================================================================== CHẠY
def chay(mau: str, kich_ban: str = "chuan", that: bool = False, in_trace: bool = True) -> dict:
    global _kb
    _kb = KICH_BAN[kich_ban]
    _dat_cho.clear()
    rb = RangBuoc()
    h = Harness(rb, ngan_sach=_kb.get("ngan_sach", 12), in_trace=in_trace)
    model = _model_that() if that else ModelGia()

    h.log("=" * 90 + f"\n{mau.upper()} · kịch bản {kich_ban} · {rb.cau_hoi()}\n" + "=" * 90)
    if mau == "react":
        agent = create_agent(model=model, tools=TOOLS_LC, system_prompt=PROMPT_REACT,
                             middleware=[HarnessMiddleware(h)])
    else:
        agent = tao_plan_graph(model, h, cho_lap_lai=(mau == "lai"))
    try:
        agent.invoke({"messages": [HumanMessage(rb.cau_hoi())]}, {"recursion_limit": 60})
    except GraphRecursionError as e:                     # lớp cuối: trần cứng của LangGraph
        h._dung("Hết ngân sách", f"RECURSION · {type(e).__name__}", "Tăng giới hạn hay dừng?")

    kq = h.ket_qua or h.ket_thuc_model("")
    h.log("-" * 90 + f"\n[{kq['kieu']}]\n{kq['noi_dung']}")
    ve_sai = [b["code"] for b in _dat_cho.values() if b["paid"] and rb.vi_pham(b)]
    return {"kich_ban": kich_ban, "mau": mau, "kieu": kq["kieu"], "dat": kq["kieu"] == "Đạt mục tiêu",
            "goi_model": h.goi_model, "goi_tool": h.goi_tool, "bi_chan": h.bi_chan, "token": h.token,
            "ve_sai": len(ve_sai), "giu_cho_treo": sum(not b["paid"] for b in _dat_cho.values())}


def danh_gia() -> list:
    hang = [chay(mau, kb, in_trace=False) for kb in KICH_BAN for mau in ("react", "plan", "lai")]
    print("| Kịch bản | Mẫu | Kết thúc | Gọi model | Gọi tool | Harness chặn | Token ~ | Vé sai lọt | Giữ chỗ treo |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in hang:
        print(f"| {r['kich_ban']} | {r['mau']} | {r['kieu']} | {r['goi_model']} | {r['goi_tool']} | "
              f"{r['bi_chan']} | {r['token']} | {r['ve_sai']} | {r['giu_cho_treo']} |")

    print("\n| Mẫu | Đạt mục tiêu (chuan + doi_gia) | Tổng gọi model | Tổng token ~ | Tổng lần harness chặn |")
    print("|---|---:|---:|---:|---:|")
    for mau in ("react", "plan", "lai"):
        rs = [r for r in hang if r["mau"] == mau]
        dat = sum(r["dat"] for r in rs if r["kich_ban"] in ("chuan", "doi_gia"))
        print(f"| {mau} | {dat}/2 | {sum(r['goi_model'] for r in rs)} | "
              f"{sum(r['token'] for r in rs)} | {sum(r['bi_chan'] for r in rs)} |")

    # tự kiểm: các bất biến harness phải giữ với mọi mẫu
    assert all(r["ve_sai"] == 0 for r in hang), "có vé vi phạm ràng buộc lọt qua harness"
    assert all(r["dat"] for r in hang if r["kich_ban"] == "chuan")
    assert all(r["kieu"] == "Cần con người" for r in hang if r["kich_ban"] == "can_duyet")
    assert all(not r["dat"] for r in hang if r["kich_ban"] in ("loi_dich_vu", "ngan_sach"))
    print("\nTự kiểm: OK")
    return hang


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="BTVN#3 · agent đặt vé máy bay")
    p.add_argument("--mau", default="react", choices=["react", "plan", "lai"])
    p.add_argument("--kich-ban", default="chuan", choices=list(KICH_BAN))
    p.add_argument("--danh-gia", action="store_true", help="chạy 5 kịch bản × 3 mẫu, in bảng so sánh")
    p.add_argument("--that", action="store_true", help="dùng model thật (SE373_MODEL)")
    a = p.parse_args()
    if a.danh_gia:
        danh_gia()
    else:
        print(json.dumps(chay(a.mau, a.kich_ban, a.that), ensure_ascii=False))


if __name__ == "__main__":
    main()
