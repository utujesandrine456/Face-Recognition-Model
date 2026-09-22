# src/camera.py
"""
Preview a camera by OpenCV index.

  python -m src.camera          # default index 0
  python -m src.camera --cam 1  # USB / embedded camera (try 1 or 2)
  python -m src.camera --list   # probe which indices open
"""
from __future__ import annotations

import argparse

import cv2


def list_cameras(max_index: int = 6) -> None:
    print("Probing camera indices (CAP_DSHOW)...")
    found = False
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"  {i}: closed")
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  {i}: OK  {w}x{h}")
            found = True
        else:
            print(f"  {i}: opened but no frame")
        cap.release()
    if not found:
        print("No cameras found.")


def main():
    parser = argparse.ArgumentParser(description="Camera preview / list")
    parser.add_argument("--cam", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--list", action="store_true", help="List working camera indices")
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Camera index {args.cam} not opened. Try: python -m src.camera --list")

    print(f"Camera test index={args.cam}. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        cv2.putText(frame, f"cam={args.cam}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Camera Test", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
