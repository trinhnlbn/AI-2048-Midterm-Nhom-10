"""Agent 2048: Expectimax + Iterative Deepening + Transposition Table.

Ý tưởng chính
-------------
- Trong lúc search, bàn cờ được biểu diễn bằng một số nguyên 64-bit
  (16 ô x 4 bit, mỗi ô lưu số mũ: ô 2 -> 1, ô 4 -> 2, ...). Mọi nước đi và
  hàm heuristic dùng bảng tra cứu theo từng hàng (65536 hàng) nên rất nhanh
  trong Python thuần.
- Ở gốc (root), nước đi được tính bằng ``move_board`` của engine nên không
  phụ thuộc vào cách đặt tên/hướng của ``Move``.
- Iterative deepening theo ngân sách thời gian của từng nước, ngân sách này
  được chia từ ``time_remaining_ms`` của cả ván và tăng khi bàn cờ chật.
- Thuật toán không dùng ngẫu nhiên nên cùng trạng thái + cùng ngân sách
  thời gian sẽ cho cùng nước đi.
"""

from __future__ import annotations

import time  

import numpy as np 

from game2048.observation import Observation  
from game2048.rules import move_board         
from game2048.types import Move              

# ---------------------------------------------------------------------------
# Bảng tra cứu theo hàng (xây một lần khi import, ~0.5 giây)
# ---------------------------------------------------------------------------
# Mỗi ô trên bàn cờ dùng 4 bit (biểu diễn được số từ 0 đến 15).
# Ta lưu SỐ MŨ của ô: ô trống -> 0, ô 2 -> 1, ô 4 -> 2, ..., ô 32768 -> 15.
_MAX_EXP = 15

# Các trọng số (hệ số điểm) dùng để AI tự chấm điểm xem một thế cờ "đẹp" hay "xấu"
_LOST_PENALTY = 200000.0  # Điểm nền cộng thêm để thế cờ còn sống luôn có điểm dương (thua = 0)
_EMPTY_WEIGHT = 270.0     # Càng nhiều ô trống càng tốt -> thưởng 270 điểm mỗi ô trống
_MERGE_WEIGHT = 700.0     # Hai ô giống nhau nằm cạnh nhau sắp gộp được -> thưởng 700 điểm
_MONO_WEIGHT = 47.0       # Phạt nặng nếu các ô trên hàng không xếp tăng dần hoặc giảm dần
_MONO_POWER = 4           # Lũy thừa bậc 4 để phạt cực nặng nếu ô lớn bị kẹt giữa các ô nhỏ
_SUM_WEIGHT = 11.0        # Phạt nhẹ theo tổng giá trị ô để ép AI gộp ô nhỏ thành ô lớn
_SUM_POWER = 3.5          # Lũy thừa tính tổng giá trị các ô


def _slide_left(line):
    """Mô phỏng thao tác gạt 1 hàng 4 ô sang bên trái (VD: [1, 0, 1, 2] -> [2, 2, 0, 0])."""
    tiles = [x for x in line if x]  # Bước 1: Bỏ hết số 0 (ô trống) đi, dồn các ô có số lại
    out = []                        # Danh sách chứa kết quả sau khi gộp
    i = 0                           # Con trỏ duyệt qua từng ô
    n = len(tiles)                  # Số lượng ô khác 0
    while i < n:
        # Nếu 2 ô đứng cạnh nhau có cùng giá trị và chưa chạm trần 15 (32768):
        if i + 1 < n and tiles[i] == tiles[i + 1] and tiles[i] < _MAX_EXP:
            out.append(tiles[i] + 1)  # Gộp 2 ô lại -> số mũ tăng thêm 1 (VD: 2+2=4 -> mũ 1+1=2)
            i += 2                    # Nhảy qua cả 2 ô vừa gộp
        else:
            out.append(tiles[i])      # Không gộp được thì giữ nguyên ô hiện tại
            i += 1                    # Chuyển sang xét ô tiếp theo
    out += [0] * (4 - len(out))       # Bước 2: Bù thêm số 0 vào bên phải cho đủ 4 ô
    return out


def _enc(line):
    """Nén 4 số (mỗi số 4 bit) của 1 hàng thành 1 số nguyên 16-bit duy nhất (0 -> 65535)."""
    # Dịch trái (<<) từng ô đúng vị trí 4-bit của nó rồi ghép lại bằng phép OR (|)
    return line[0] | (line[1] << 4) | (line[2] << 8) | (line[3] << 12)


def _row_heuristic(line):
    """Chấm điểm độ đẹp của 1 hàng 4 ô dựa trên 4 tiêu chí của cao thủ 2048."""
    s = 0.0        # Tổng lũy thừa các ô (dùng để phạt nếu bàn cờ quá nhiều ô rác)
    empty = 0      # Đếm số ô trống trên hàng
    merges = 0     # Đếm số cặp ô giống nhau có thể gộp được
    prev = 0       # Lưu giá trị ô khác 0 gần nhất bên trái
    counter = 0    # Đếm chuỗi các ô giống nhau liên tiếp
    for rank in line:
        s += rank ** _SUM_POWER          # Cộng dồn giá trị ô để tính điểm phạt tổng
        if rank == 0:
            empty += 1                   # Gặp ô trống -> tăng biến đếm
        else:
            if prev == rank:             # Nếu ô này giống hệt ô khác 0 liền trước
                counter += 1             # Ghi nhận thêm 1 cặp có thể gộp
            elif counter > 0:            # Nếu chuỗi giống nhau bị ngắt bởi ô khác số
                merges += 1 + counter    # Cộng điểm thưởng gộp
                counter = 0
            prev = rank                  # Cập nhật ô khác 0 gần nhất
    if counter > 0:                      # Nếu chuỗi giống nhau nằm ở cuối hàng
        merges += 1 + counter

    # Kiểm tra tính đơn điệu (các ô nên xếp bậc thang: tăng dần hoặc giảm dần)
    mono_l = 0.0   # Điểm phạt nếu vi phạm chiều tăng dần (trái nhỏ -> phải lớn)
    mono_r = 0.0   # Điểm phạt nếu vi phạm chiều giảm dần (trái lớn -> phải nhỏ)
    for i in range(1, 4):
        a, b = line[i - 1], line[i]      # Xét 2 ô cạnh nhau: a (trái) và b (phải)
        if a > b:
            # Trái lớn hơn phải -> sai trật tự tăng dần -> cộng điểm phạt vào mono_l
            mono_l += a ** _MONO_POWER - b ** _MONO_POWER
        else:
            # Trái nhỏ hơn hoặc bằng phải -> sai trật tự giảm dần -> cộng điểm phạt vào mono_r
            mono_r += b ** _MONO_POWER - a ** _MONO_POWER

    # Công thức tổng hợp điểm: Nền + Ô trống + Khả năng gộp - Phạt lộn xộn - Phạt tổng ô
    return (
        _LOST_PENALTY
        + _EMPTY_WEIGHT * empty
        + _MERGE_WEIGHT * merges
        - _MONO_WEIGHT * min(mono_l, mono_r)  # Chỉ cần tăng dần HOẶC giảm dần đều được
        - _SUM_WEIGHT * s
    )


def _build_tables():
    """Tính trước kết quả cho tất cả 65,536 trạng thái của 1 hàng để khi chơi chỉ việc tra sổ."""
    left = [0] * 65536     # Bảng lưu kết quả khi gạt sang trái
    right = [0] * 65536    # Bảng lưu kết quả khi gạt sang phải
    heur = [0.0] * 65536   # Bảng lưu điểm heuristic của từng kiểu hàng
    for r in range(65536):
        # Tách số nguyên 16-bit r thành 4 ô riêng lẻ [ô0, ô1, ô2, ô3]
        line = [(r >> (4 * i)) & 0xF for i in range(4)]
        left[r] = _enc(_slide_left(line))                # Tính kết quả gạt trái
        # Gạt phải = lật ngược hàng (line[::-1]), gạt trái, rồi lật ngược lại
        right[r] = _enc(_slide_left(line[::-1])[::-1])
        heur[r] = _row_heuristic(line)                   # Tính điểm đẹp của hàng này
    return left, right, heur


# Khởi tạo ngay 3 bảng tra cứu khi nạp file (chỉ chạy đúng 1 lần duy nhất)
_LEFT, _RIGHT, _HEUR = _build_tables()


def _transpose(x: int) -> int:
    """Xoay chéo bàn cờ 64-bit (biến 4 cột dọc thành 4 hàng ngang và ngược lại)."""
    a1 = x & 0xF0F00F0FF0F00F0F
    a2 = x & 0x0000F0F00000F0F0
    a3 = x & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


def _apply_table(bb: int, tab) -> int:
    """Cắt bàn cờ 64-bit thành 4 hàng ngang, tra bảng tab cho từng hàng rồi ghép lại."""
    return (
        tab[bb & 0xFFFF]                      # Tra bảng cho hàng 0 (16 bit cuối)
        | (tab[(bb >> 16) & 0xFFFF] << 16)    # Tra bảng cho hàng 1, đẩy về đúng vị trí
        | (tab[(bb >> 32) & 0xFFFF] << 32)    # Tra bảng cho hàng 2, đẩy về đúng vị trí
        | (tab[bb >> 48] << 48)               # Tra bảng cho hàng 3 (16 bit đầu)
    )


def _move(bb: int, d: int) -> int:
    """Di chuyển toàn bộ bàn cờ 64-bit theo hướng d (0: Trái, 1: Phải, 2: Lên, 3: Xuống)."""
    if d == 0:
        return _apply_table(bb, _LEFT)        # Trái: tra trực tiếp bảng _LEFT
    if d == 1:
        return _apply_table(bb, _RIGHT)       # Phải: tra trực tiếp bảng _RIGHT
    # Lên hoặc Xuống: xoay chéo bàn cờ cho cột thành hàng -> gạt trái/phải -> xoay trả lại
    t = _transpose(bb)
    t = _apply_table(t, _LEFT if d == 2 else _RIGHT)
    return _transpose(t)


def _heuristic(bb: int) -> float:
    """Chấm điểm toàn bộ bàn cờ = Tổng điểm 4 hàng ngang + Tổng điểm 4 cột dọc."""
    t = _transpose(bb)  # Xoay chéo bàn cờ để lấy 4 cột dọc
    h = _HEUR
    return (
        h[bb & 0xFFFF] + h[(bb >> 16) & 0xFFFF] + h[(bb >> 32) & 0xFFFF] + h[bb >> 48]
        + h[t & 0xFFFF] + h[(t >> 16) & 0xFFFF] + h[(t >> 32) & 0xFFFF] + h[t >> 48]
    )


class _Timeout(Exception):
    """Tín hiệu báo hết giờ dùng để thoát khẩn cấp khỏi vòng lặp tìm kiếm đệ quy."""
    pass


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent:
    name = "Nhom 10"

    # --- Các tham số cấu hình cho AI ---------------------------------------
    MAX_DEPTH = 12             
    CPROB_CUTOFF = 1e-4        
    EST_TOTAL_MOVES = 4000     
    MIN_MOVES_LEFT = 400       
    MAX_TT_SIZE = 1_500_000   

    def __init__(self) -> None:
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        """Xóa sạch trí nhớ tạm và bộ đếm khi bắt đầu một ván mới."""
        self._tt: dict[int, tuple[int, float]] = {}  # Bảng băm lưu {bàn_cờ: (độ_sâu, điểm)}
        self._moves = 0                              # Đếm số nước đã đi từ đầu ván
        self._exp_mode = False                       # True nếu game truyền sẵn số mũ (1,2,3...)
        self._deadline: float | None = None          # Mốc thời gian bắt buộc phải dừng suy nghĩ
        self._nodes = 0                              # Đếm số thế cờ đã duyệt qua trong lượt hiện tại

    # ------------------------------------------------------------------
    # Chuyển đổi board numpy -> bitboard
    # ------------------------------------------------------------------
    def _to_bb(self, board: np.ndarray) -> int:
        """Nén bàn cờ mảng 4x4 của game thành 1 số nguyên 64-bit duy nhất."""
        bb = 0
        exp_mode = self._exp_mode
        for i, v in enumerate(board.ravel().tolist()):  # Trải phẳng 16 ô thành danh sách 1 chiều
            v = int(v)
            if v:
                # Nếu game dùng số mũ (1,2,3) thì lấy luôn; nếu dùng giá trị (2,4,8) thì lấy log2
                e = v if exp_mode else v.bit_length() - 1
                if e > _MAX_EXP:
                    e = _MAX_EXP                        # Kẹp tối đa ở mức 15
                bb |= e << (4 * i)                      # Nhét ô thứ i vào đúng vị trí 4-bit của nó
        return bb

    # ------------------------------------------------------------------
    # Expectimax (Thuật toán nhìn trước tương lai)
    # ------------------------------------------------------------------
    def _max_node(self, bb: int, depth: int, cprob: float) -> float:
        """Lượt của AI: Chọn hướng đi mang lại điểm kỳ vọng cao nhất."""
        # Điểm dừng: Đã nhìn đủ độ sâu HOẶC nhánh này quá hiếm gặp -> chấm điểm luôn
        if depth == 0 or cprob < self.CPROB_CUTOFF:
            return _heuristic(bb)

        # Cứ duyệt qua 256 thế cờ thì kiểm tra đồng hồ 1 lần; quá giờ thì ngắt ngay
        self._nodes += 1
        if self._deadline is not None and not (self._nodes & 255):
            if time.perf_counter() > self._deadline:
                raise _Timeout

        # Kiểm tra trí nhớ tạm: Nếu thế cờ này đã tính ở độ sâu >= hiện tại thì lấy luôn kết quả
        ent = self._tt.get(bb)
        if ent is not None and ent[0] >= depth:
            return ent[1]

        best = 0.0  # Nếu không có nước đi nào hợp lệ (thua cuộc), hàm sẽ trả về 0.0
        for d in range(4):                          # Thử lần lượt 4 hướng: Trái, Phải, Lên, Xuống
            nb = _move(bb, d)
            if nb == bb:                            # Nếu gạt hướng này mà bàn cờ đứng yên -> bỏ qua
                continue
            v = self._chance_node(nb, depth, cprob) # Hỏi tiếp xem máy sẽ đẻ ô mới thế nào
            if v > best:
                best = v                            # Giữ lại hướng đi có điểm cao nhất

        self._tt[bb] = (depth, best)                # Ghi nhớ kết quả thế cờ này vào bộ nhớ đệm
        return best

    def _chance_node(self, bb: int, depth: int, cprob: float) -> float:
        """Lượt của Máy: Tính điểm trung bình của mọi khả năng sinh ô 2 (90%) hoặc ô 4 (10%)."""
        # Tìm vị trí bit (shift) của tất cả các ô đang trống trên bàn cờ
        empties = []
        for shift in range(0, 64, 4):
            if not (bb >> shift) & 0xF:
                empties.append(shift)
        n = len(empties)
        if n == 0:                                  # Không còn ô trống nào -> chấm điểm luôn
            return _heuristic(bb)

        cp = cprob / n                              # Chia đều xác suất cho n ô trống
        total = 0.0
        for shift in empties:
            # Khả năng 1 (90%): Máy sinh ô số 2 (mũ 1) tại ô trống này, giảm depth đi 1
            total += 0.9 * self._max_node(bb | (1 << shift), depth - 1, cp * 0.9)
            # Khả năng 2 (10%): Máy sinh ô số 4 (mũ 2) tại ô trống này, giảm depth đi 1
            total += 0.1 * self._max_node(bb | (2 << shift), depth - 1, cp * 0.1)
        return total / n                            # Trả về điểm trung bình cộng

    # ------------------------------------------------------------------
    # Quản lý thời gian
    # ------------------------------------------------------------------
    def _budget_sec(self, remaining_ms: float, empties: int) -> float:
        """Tính số giây được phép suy nghĩ cho nước đi hiện tại dựa trên độ nguy hiểm của bàn cờ."""
        moves_left = max(self.MIN_MOVES_LEFT, self.EST_TOTAL_MOVES - self._moves)
        base_ms = remaining_ms / moves_left         # Thời gian trung bình cho 1 nước

        # Bàn càng ít ô trống (càng sắp thua) thì cấp càng nhiều thời gian suy nghĩ
        if empties >= 9:
            factor = 0.35      # Bàn rất thoáng -> đi nhanh
        elif empties >= 6:
            factor = 0.8
        elif empties >= 4:
            factor = 1.6
        else:
            factor = 3.5       # Bàn cực chật (< 4 ô trống) -> suy nghĩ kỹ gấp 3.5 lần

        budget_ms = base_ms * factor
        # Không bao giờ tiêu quá 10% tổng thời gian còn lại và tối đa 2.5 giây/nước
        budget_ms = min(budget_ms, remaining_ms * 0.10, 2500.0)
        if remaining_ms < 1500:                     # Tổng giờ còn dưới 1.5s -> đi siêu tốc (<= 2ms)
            budget_ms = min(budget_ms, 2.0)
        return max(budget_ms, 1.0) / 1000.0         # Đổi từ mili-giây sang giây

    # ------------------------------------------------------------------
    # Chọn nước đi
    # ------------------------------------------------------------------
    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        """Hàm chính được hệ thống game gọi mỗi lượt để lấy nước đi."""
        legal = list(obs.legal_moves)
        if not legal:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")

        self._moves += 1
        if len(legal) == 1:                         # Chỉ có 1 hướng đi duy nhất -> đi luôn khỏi nghĩ
            return legal[0]

        try:
            move = self._search(obs, legal, time_limit_ms)
        except Exception:                           # Đề phòng mọi lỗi bất ngờ để không bị xử thua
            move = legal[0]
        return move if move in legal else legal[0]

    def _search(self, obs: Observation, legal: list, time_limit_ms: float) -> Move:
        """Tìm kiếm sâu dần (Iterative Deepening): tìm độ sâu 1, 2, 3... cho đến khi hết giờ."""
        board = np.asarray(obs.board)

        # Tự nhận diện xem game truyền số mũ (có số lẻ như 1, 3) hay giá trị thật (toàn số chẵn)
        if not self._exp_mode and bool((board.astype(np.int64) & 1).any()):
            self._exp_mode = True

        # Xóa bớt bộ nhớ đệm nếu đã lưu quá 1.5 triệu thế cờ
        if len(self._tt) > self.MAX_TT_SIZE:
            self._tt.clear()

        # Lấy tổng thời gian còn lại và tính ngân sách thời gian (budget) cho nước này
        remaining = getattr(obs, "time_remaining_ms", None)
        remaining = float(time_limit_ms if remaining is None else remaining)
        if time_limit_ms:
            remaining = min(remaining, float(time_limit_ms))

        empties = int(np.count_nonzero(board == 0))
        budget = self._budget_sec(remaining, empties)

        # Dùng hàm chuẩn của game để tạo bàn cờ sau nước đi đầu tiên cho từng hướng hợp lệ
        cands = []
        for mv in legal:
            nb, _score, _changed = move_board(obs.board, mv)
            cands.append((mv, self._to_bb(np.asarray(nb))))

        start = time.perf_counter()
        best_idx = 0

        # Tăng dần độ sâu tìm kiếm từ 1 lên tối đa 12
        for depth in range(1, self.MAX_DEPTH + 1):
            t0 = time.perf_counter()
            # Độ sâu 1 không đặt giới hạn giờ để đảm bảo luôn có ít nhất 1 đáp án hoàn chỉnh
            self._deadline = None if depth == 1 else start + budget * 1.5
            self._nodes = 0
            try:
                # Chấm điểm tất cả các nước ứng viên ở độ sâu hiện tại
                vals = [self._chance_node(bb, depth, 1.0) for _mv, bb in cands]
            except _Timeout:
                # Nếu đang tính dở mà hết giờ -> thoát ngay và giữ đáp án từ độ sâu trước đó
                break
            finally:
                self._deadline = None

            # Cập nhật nước đi tốt nhất vừa tìm được ở độ sâu này
            best_val = vals[0]
            best_idx = 0
            for i in range(1, len(vals)):
                if vals[i] > best_val:
                    best_val = vals[i]
                    best_idx = i

            now = time.perf_counter()
            # Độ sâu tiếp theo thường tốn gấp ~4 lần độ sâu vừa rồi (now - t0).
            # Nếu ước tính không đủ giờ làm trọn vòng tiếp theo thì chủ động dừng sớm luôn.
            if (now - start) + 4.0 * (now - t0) > budget:
                break

        return cands[best_idx][0]