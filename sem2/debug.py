#!/usr/bin/env python3
"""
debug_freq.py — 独立频率调试脚本
========================================
每隔 1 秒打印 4 个 parking 传感器的原始频率，
完全绕开 EMA / 中位数 / 基线逻辑，直接看硬件在读什么。

运行方式:
  sudo python3 debug_freq.py

退出: Ctrl+C
"""

import pigpio
import time

# ── 配置（与 program.py 保持一致）──────────────────────────
PARK_GPIOS   = [17, 27, 22, 23]
PARK_COUNT   = len(PARK_GPIOS)
SAMPLE_S     = 1.0   # 打印间隔（秒）

# ── 状态 ────────────────────────────────────────────────────
_last_tick  = [None] * PARK_COUNT
_period_sum = [0.0]  * PARK_COUNT   # 采样窗口内所有 period 之和 (µs)
_period_cnt = [0]    * PARK_COUNT   # 本窗口内有效边沿数

# ── 回调：记录每个上升沿的原始周期 ──────────────────────────
def make_cb(idx):
    def cb(gpio, level, tick):
        last = _last_tick[idx]
        _last_tick[idx] = tick
        if last is None:
            return
        period = pigpio.tickDiff(last, tick)
        if period <= 0:
            return
        _period_sum[idx] += period
        _period_cnt[idx] += 1
    return cb

# ── 主程序 ───────────────────────────────────────────────────
pi = pigpio.pi()
if not pi.connected:
    print("[ERROR] pigpiod 没有运行，请先执行: sudo systemctl start pigpiod")
    exit(1)

print(f"[DEBUG] 监听 GPIO {PARK_GPIOS}，每 {SAMPLE_S:.0f} 秒打印一次\n")
print(f"{'时间':10s}  {'P1':>12s}  {'P2':>12s}  {'P3':>12s}  {'P4':>12s}  {'边沿数(P1-P4)'}")
print("-" * 80)

for idx, gpio in enumerate(PARK_GPIOS):
    pi.set_mode(gpio, pigpio.INPUT)
    pi.set_pull_up_down(gpio, pigpio.PUD_OFF)
    pi.callback(gpio, pigpio.RISING_EDGE, make_cb(idx))

try:
    while True:
        time.sleep(SAMPLE_S)

        now = time.strftime("%H:%M:%S")
        freqs  = []
        counts = []

        for i in range(PARK_COUNT):
            cnt = _period_cnt[i]
            s   = _period_sum[i]

            # 重置窗口
            _period_sum[i] = 0.0
            _period_cnt[i] = 0

            if cnt > 0:
                avg_period = s / cnt          # µs
                freq = 1_000_000.0 / avg_period
            else:
                freq = 0.0

            freqs.append(freq)
            counts.append(cnt)

        freq_strs  = [f"{f:>10.1f} Hz" for f in freqs]
        count_strs = "  ".join(str(c) for c in counts)

        print(f"{now:10s}  {'  '.join(freq_strs)}  [{count_strs}]")

except KeyboardInterrupt:
    print("\n[DEBUG] 停止")
    pi.stop()