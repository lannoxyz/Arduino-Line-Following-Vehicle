"""
collect_data.py — 树莓派数据采集脚本
用法: python3 collect_data.py

操作说明:
  把形状放在摄像头前
  按 SPACE  → 拍一张
  按 A      → 自动连拍 (每0.5秒一张，共20张)
  按 N      → 下一个类别
  按 B      → 返回上一个类别
  按 Q      → 退出并显示汇总

采集完成后把整个 dataset/ 文件夹复制到你的电脑上训练。
"""

import cv2, os, time
from picamera2 import Picamera2

# ── 配置 ──────────────────────────────────────────
CLASSES = [
    "Star",
    "Octagon",
    "Cross",
    "Trapezium",
    "Diamond",
    "CuttedCircle",
    "QuarterCircle",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "RecycleSign",
    "ButtonSign",
    "WarningSign",
]
SAVE_DIR   = "dataset"
RESOLUTION = (640, 480)
FLIP_MODE  = -1
AUTO_COUNT = 20      # 自动连拍张数
AUTO_DELAY = 0.5     # 连拍间隔(秒)
# ──────────────────────────────────────────────────

def count_imgs(folder):
    if not os.path.exists(folder): return 0
    return len([f for f in os.listdir(folder) if f.endswith(".jpg")])

def print_summary():
    print("\n=== 采集汇总 ===")
    total = 0
    for cls in CLASSES:
        folder = os.path.join(SAVE_DIR, cls)
        n = count_imgs(folder)
        total += n
        bar   = "█" * (n // 5)
        warn  = "  ⚠ 建议补充" if n < 30 else ""
        print(f"  {cls:<16} {n:>4} 张  {bar}{warn}")
    print(f"\n  总计: {total} 张")

# 初始化摄像头
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": RESOLUTION, "format": "BGR888"}))
picam2.start()
time.sleep(1)

cls_idx = 0

print("\n=== 数据采集开始 ===")
print(f"共 {len(CLASSES)} 个类别: {', '.join(CLASSES)}")
print(f"保存路径: {SAVE_DIR}/\n")

while 0 <= cls_idx < len(CLASSES):
    cls_name    = CLASSES[cls_idx]
    save_folder = os.path.join(SAVE_DIR, cls_name)
    os.makedirs(save_folder, exist_ok=True)

    print(f"\n>>> [{cls_idx+1}/{len(CLASSES)}] 当前类别: {cls_name}  已采集: {count_imgs(save_folder)} 张")
    print("  SPACE=拍一张  A=自动连拍  N=下一类  B=上一类  Q=退出")

    while True:
        frame = picam2.capture_array()
        if FLIP_MODE is not None:
            frame = cv2.flip(frame, FLIP_MODE)

        disp = frame.copy()
        n    = count_imgs(save_folder)

        # 顶部状态栏
        cv2.rectangle(disp, (0, 0), (640, 55), (0, 0, 0), -1)
        cv2.putText(disp, f"[{cls_idx+1}/{len(CLASSES)}]  {cls_name}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2)
        cv2.putText(disp, f"{n} imgs captured",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        # 进度条
        progress = cls_idx / len(CLASSES)
        cv2.rectangle(disp, (0, 55), (640, 60), (40, 40, 40), -1)
        cv2.rectangle(disp, (0, 55), (int(640 * progress), 60), (0, 200, 100), -1)

        # 中心构图框
        h, w = frame.shape[:2]
        cx, cy, sz = w//2, h//2, 180
        cv2.rectangle(disp, (cx-sz, cy-sz), (cx+sz, cy+sz), (0, 200, 255), 2)
        cv2.putText(disp, "place shape here",
                    (cx-sz+6, cy-sz+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # 底部操作提示
        cv2.rectangle(disp, (0, 455), (640, 480), (0, 0, 0), -1)
        cv2.putText(disp, "SPACE=snap  A=auto  N=next  B=back  Q=quit",
                    (10, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

        # 数量提示
        if n < 30:
            cv2.putText(disp, f"Need more! ({n}/30 min)",
                        (w-200, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)
        else:
            cv2.putText(disp, f"OK ({n} imgs)",
                        (w-130, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)

        cv2.imshow("Collect Data", disp)
        key = cv2.waitKey(1) & 0xFF

        def save_frame(f):
            ts   = int(time.time() * 1000)
            path = os.path.join(save_folder, f"{cls_name}_{ts}.jpg")
            cv2.imwrite(path, f)
            return path

        if key == ord('q'):
            print("\n已退出采集。")
            cv2.destroyAllWindows()
            picam2.stop()
            print_summary()
            exit(0)

        elif key == ord(' '):
            p = save_frame(frame)
            print(f"  保存: {os.path.basename(p)}  (共{count_imgs(save_folder)}张)")

        elif key == ord('a'):
            print(f"  自动连拍 {AUTO_COUNT} 张...")
            for i in range(AUTO_COUNT):
                f = picam2.capture_array()
                if FLIP_MODE is not None:
                    f = cv2.flip(f, FLIP_MODE)
                save_frame(f)
                prog_disp = frame.copy()
                cv2.rectangle(prog_disp, (0,0),(640,55),(0,0,0),-1)
                cv2.putText(prog_disp, f"Auto shooting: {i+1}/{AUTO_COUNT}",
                            (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.imshow("Collect Data", prog_disp)
                cv2.waitKey(1)
                time.sleep(AUTO_DELAY)
            print(f"  连拍完成！共 {count_imgs(save_folder)} 张")

        elif key == ord('n'):
            print(f"  [{cls_name}] 完成，共 {count_imgs(save_folder)} 张")
            cls_idx += 1
            break

        elif key == ord('b'):
            print(f"  返回上一类别")
            cls_idx = max(0, cls_idx - 1)
            break

cv2.destroyAllWindows()
picam2.stop()
print_summary()
print(f"\n✓ 数据集保存在 ./{SAVE_DIR}/")
print("  请将 dataset/ 文件夹复制到电脑，然后运行 train.py")
