"""Agent 2048: Expectimax + Iterative Deepening + Transposition Table.

Ý tưởng chính
-------------
* Trong lúc search, bàn cờ được biểu diễn bằng một số nguyên 64-bit
  (16 ô x 4 bit, mỗi ô lưu số mũ: ô 2 -> 1, ô 4 -> 2, ...). Mọi nước đi và
  hàm heuristic dùng bảng tra cứu theo từng hàng (65536 hàng) nên rất nhanh
  trong Python thuần.
* Ở gốc (root), nước đi được tính bằng ``move_board`` của engine nên không
  phụ thuộc vào cách đặt tên/hướng của ``Move``.
* Iterative deepening theo ngân sách thời gian của từng nước, ngân sách này
  được chia từ ``time_remaining_ms`` của cả ván và tăng khi bàn cờ chật.
* Thuật toán không dùng ngẫu nhiên nên cùng trạng thái + cùng ngân sách
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
_MAX_EXP = 15  # nibble tối đa; ô >= 32768 bị kẹp về 15 trong lúc search

_LOST_PENALTY = 200000.0
_EMPTY_WEIGHT = 270.0
_MERGE_WEIGHT = 700.0
_MONO_WEIGHT = 47.0
_MONO_POWER = 4
_SUM_WEIGHT = 11.0
_SUM_POWER = 3.5


def _slide_left(line):
    tiles = [x for x in line if x]
    out = []
    i = 0
    n = len(tiles)
    while i < n:
        if i + 1 < n and tiles[i] == tiles[i + 1] and tiles[i] < _MAX_EXP:
            out.append(tiles[i] + 1)
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (4 - len(out))
    return out


def _enc(line):
    return line[0] | (line[1] << 4) | (line[2] << 8) | (line[3] << 12)


def _row_heuristic(line):
    s = 0.0
    empty = 0
    merges = 0
    prev = 0
    counter = 0
    for rank in line:
        s += rank ** _SUM_POWER
        if rank == 0:
            empty += 1
        else:
            if prev == rank:
                counter += 1
            elif counter > 0:
                merges += 1 + counter
                counter = 0
            prev = rank
    if counter > 0:
        merges += 1 + counter

    mono_l = 0.0
    mono_r = 0.0
    for i in range(1, 4):
        a, b = line[i - 1], line[i]
        if a > b:
            mono_l += a ** _MONO_POWER - b ** _MONO_POWER
        else:
            mono_r += b ** _MONO_POWER - a ** _MONO_POWER

    return (
        _LOST_PENALTY
        + _EMPTY_WEIGHT * empty
        + _MERGE_WEIGHT * merges
        - _MONO_WEIGHT * min(mono_l, mono_r)
        - _SUM_WEIGHT * s
    )


def _build_tables():
    left = [0] * 65536
    right = [0] * 65536
    heur = [0.0] * 65536
    for r in range(65536):
        line = [(r >> (4 * i)) & 0xF for i in range(4)]
        left[r] = _enc(_slide_left(line))
        right[r] = _enc(_slide_left(line[::-1])[::-1])
        heur[r] = _row_heuristic(line)
    return left, right, heur


_LEFT, _RIGHT, _HEUR = _build_tables()


def _transpose(x: int) -> int:
    a1 = x & 0xF0F00F0FF0F00F0F
    a2 = x & 0x0000F0F00000F0F0
    a3 = x & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


def _apply_table(bb: int, tab) -> int:
    return (
        tab[bb & 0xFFFF]
        | (tab[(bb >> 16) & 0xFFFF] << 16)
        | (tab[(bb >> 32) & 0xFFFF] << 32)
        | (tab[bb >> 48] << 48)
    )


def _move(bb: int, d: int) -> int:
    if d == 0:
        return _apply_table(bb, _LEFT)
    if d == 1:
        return _apply_table(bb, _RIGHT)
    t = _transpose(bb)
    t = _apply_table(t, _LEFT if d == 2 else _RIGHT)
    return _transpose(t)


def _heuristic(bb: int) -> float:
    t = _transpose(bb)
    h = _HEUR
    return (
        h[bb & 0xFFFF]
        + h[(bb >> 16) & 0xFFFF]
        + h[(bb >> 32) & 0xFFFF]
        + h[bb >> 48]
        + h[t & 0xFFFF]
        + h[(t >> 16) & 0xFFFF]
        + h[(t >> 32) & 0xFFFF]
        + h[t >> 48]
    )


class _Timeout(Exception):
    pass


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent:
    name = "Nhom 10"

    # --- tham số có thể chỉnh -------------------------------------------
    MAX_DEPTH = 12
    CPROB_CUTOFF = 1e-4        # bỏ nhánh có xác suất quá thấp
    EST_TOTAL_MOVES = 4000     # ước lượng số nước/ván để chia thời gian
    MIN_MOVES_LEFT = 400       # luôn giả định còn ít nhất ngần này nước
    MAX_TT_SIZE = 1_500_000

    def __init__(self) -> None:
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        """Xóa cache/trạng thái giữa hai ván."""
        self._tt: dict[int, tuple[int, float]] = {}
        self._moves = 0
        self._exp_mode = False   # True nếu engine lưu số mũ thay vì giá trị ô
        self._deadline: float | None = None
        self._nodes = 0

    # ------------------------------------------------------------------
    # Chuyển đổi board numpy -> bitboard
    # ------------------------------------------------------------------
    def _to_bb(self, board: np.ndarray) -> int:
        bb = 0
        exp_mode = self._exp_mode
        for i, v in enumerate(board.ravel().tolist()):
            v = int(v)
            if v:
                e = v if exp_mode else v.bit_length() - 1
                if e > _MAX_EXP:
                    e = _MAX_EXP
                bb |= e << (4 * i)
        return bb

    # ------------------------------------------------------------------
    # Expectimax
    # ------------------------------------------------------------------
    def _max_node(self, bb: int, depth: int, cprob: float) -> float:
        if depth == 0 or cprob < self.CPROB_CUTOFF:
            return _heuristic(bb)

        self._nodes += 1
        if self._deadline is not None and not (self._nodes & 255):
            if time.perf_counter() > self._deadline:
                raise _Timeout

        ent = self._tt.get(bb)
        if ent is not None and ent[0] >= depth:
            return ent[1]

        best = 0.0
        for d in range(4):
            nb = _move(bb, d)
            if nb == bb:
                continue
            v = self._chance_node(nb, depth, cprob)
            if v > best:
                best = v

        self._tt[bb] = (depth, best)
        return best

    def _chance_node(self, bb: int, depth: int, cprob: float) -> float:
        empties = []
        for shift in range(0, 64, 4):
            if not (bb >> shift) & 0xF:
                empties.append(shift)
        n = len(empties)
        if n == 0:
            return _heuristic(bb)

        cp = cprob / n
        total = 0.0
        for shift in empties:
            total += 0.9 * self._max_node(bb | (1 << shift), depth - 1, cp * 0.9)
            total += 0.1 * self._max_node(bb | (2 << shift), depth - 1, cp * 0.1)
        return total / n

    # ------------------------------------------------------------------
    # Quản lý thời gian
    # ------------------------------------------------------------------
    def _budget_sec(self, remaining_ms: float, empties: int) -> float:
        moves_left = max(self.MIN_MOVES_LEFT, self.EST_TOTAL_MOVES - self._moves)
        base_ms = remaining_ms / moves_left

        # Bàn trống -> đi nhanh; bàn chật -> search sâu hơn.
        if empties >= 9:
            factor = 0.35
        elif empties >= 6:
            factor = 0.8
        elif empties >= 4:
            factor = 1.6
        else:
            factor = 3.5

        budget_ms = base_ms * factor
        budget_ms = min(budget_ms, remaining_ms * 0.10, 2500.0)
        if remaining_ms < 1500:          # sắp hết giờ: đi gần như tức thì
            budget_ms = min(budget_ms, 2.0)
        return max(budget_ms, 1.0) / 1000.0

    # ------------------------------------------------------------------
    # Chọn nước đi
    # ------------------------------------------------------------------
    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        """Trả về đúng một phần tử ``Move`` có trong ``obs.legal_moves``."""
        legal = list(obs.legal_moves)
        if not legal:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")

        self._moves += 1
        if len(legal) == 1:
            return legal[0]

        try:
            move = self._search(obs, legal, time_limit_ms)
        except Exception:  # không bao giờ để lỗi làm hỏng ván
            move = legal[0]
        return move if move in legal else legal[0]

    def _search(self, obs: Observation, legal: list, time_limit_ms: float) -> Move:
        board = np.asarray(obs.board)

        # Tự phát hiện engine lưu giá trị ô (2,4,8..) hay số mũ (1,2,3..):
        # giá trị ô luôn chẵn, còn số mũ thì chắc chắn có số lẻ (vd. 1).
        if not self._exp_mode and bool((board.astype(np.int64) & 1).any()):
            self._exp_mode = True

        if len(self._tt) > self.MAX_TT_SIZE:
            self._tt.clear()

        remaining = getattr(obs, "time_remaining_ms", None)
        remaining = float(time_limit_ms if remaining is None else remaining)
        if time_limit_ms:
            remaining = min(remaining, float(time_limit_ms))

        empties = int(np.count_nonzero(board == 0))
        budget = self._budget_sec(remaining, empties)

        # Dùng move_board của engine cho các nước ở gốc.
        cands = []
        for mv in legal:
            nb, _score, _changed = move_board(obs.board, mv)
            cands.append((mv, self._to_bb(np.asarray(nb))))

        start = time.perf_counter()
        best_idx = 0

        for depth in range(1, self.MAX_DEPTH + 1):
            t0 = time.perf_counter()
            # Độ sâu 1 luôn hoàn thành; các độ sâu sau bị chặn bởi deadline cứng.
            self._deadline = None if depth == 1 else start + budget * 1.5
            self._nodes = 0
            try:
                vals = [self._chance_node(bb, depth, 1.0) for _mv, bb in cands]
            except _Timeout:
                break
            finally:
                self._deadline = None

            best_val = vals[0]
            best_idx = 0
            for i in range(1, len(vals)):
                if vals[i] > best_val:
                    best_val = vals[i]
                    best_idx = i

            now = time.perf_counter()
            if (now - start) + 4.0 * (now - t0) > budget:
                break

        return cands[best_idx][0]