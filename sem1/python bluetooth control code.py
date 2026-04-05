import serial
import tkinter as tk
from tkinter import messagebox

PORT = "COM5"   # the port
BAUD = 9600

try:
    bt = serial.Serial(PORT, BAUD, timeout=1)
    print("blueteeth connected.")
except Exception as e:
    bt = None
    print("failed.", e)


def send(cmd):
    if bt:
        bt.write(cmd.encode())
    else:
        messagebox.showwarning("Warning no blueteeth")


root = tk.Tk()
root.title("blueteeth Car Controller")
root.geometry("300x320")
root.resizable(False, False)

font = ("Arial", 16, "bold")

tk.Label(root, text="HC-05 blueteeth car", font=("Arial", 14)).pack(pady=10)


btn_forward = tk.Button(
    root,
    text="↑ Forward",
    font=font,
    width=12,
    command=lambda: send("F")
)
btn_forward.pack(pady=5)


frame = tk.Frame(root)
frame.pack()

btn_left = tk.Button(
    frame,
    text="← Left",
    font=font,
    width=8,
    command=lambda: send("L")
)
btn_left.grid(row=0, column=0, padx=5)

btn_stop = tk.Button(
    frame,
    text="■ Stop",
    font=font,
    width=8,
    command=lambda: send("S")
)
btn_stop.grid(row=0, column=1, padx=5)

btn_right = tk.Button(
    frame,
    text="Right →",
    font=font,
    width=8,
    command=lambda: send("R")
)
btn_right.grid(row=0, column=2, padx=5)


btn_backward = tk.Button(
    root,
    text="↓ Backward",
    font=font,
    width=12,
    command=lambda: send("B")
)
btn_backward.pack(pady=10)


tk.Label(root, text="lets go drifttttt 🚗✨").pack(pady=10)

root.mainloop()
