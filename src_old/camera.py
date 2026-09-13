# src/camera.py
import sys
import cv2


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # force DirectShow backend on Windows

    if not cap.isOpened():
        raise RuntimeError(f"Camera index {index} not opened. Try a different index (0/1/2).")

    print(f"Camera test on index {index}. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        cv2.imshow("Camera Test", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()