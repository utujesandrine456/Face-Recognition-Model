# face-recognition-5pt

A CPU-only face recognition pipeline: Haar detection + MediaPipe 5-point
landmarks + similarity-transform alignment + ArcFace ONNX embeddings.

See the accompanying book for full explanations. Quick start:

1. Install Python 3.11 (MediaPipe FaceMesh legacy API is not available on Python 3.13).
2. py -3.11 -m venv .venv
3. .venv\Scripts\Activate.ps1      (Windows PowerShell)
3. pip install -r requirements.txt
4. python -m src.camera            (validate webcam)
5. python -m src.detect            (validate face box)
6. python -m src.landmarks         (validate 5 points)
7. python -m src.align             (validate alignment)
8. Download the real ArcFace model into models/embedder_arcface.onnx (see book Ch.5)
9. python -m src.embed             (validate embeddings)
10. python -m src.enroll           (enroll people)
11. python -m src.evaluate         (tune threshold)
12. python -m src.recognize        (live recognition)

## Embedded (ESP8266 servo pan over MQTT)

Hardware firmware lives in `firmware/esp8266_servo_pan/`.

1. Wire servo signal to D1 (GPIO5). Power servo from external 5V. Common GND with ESP8266.
2. Flash the sketch from Arduino IDE (PubSubClient + ESP8266 Servo).
3. Open Serial Monitor at 115200: WiFi + MQTT should connect; servo self-test runs.
4. On the PC (venv active):
   - `pip install -r requirements.txt`
   - `python -m src.servo_test`  (moves servo center/left/right without camera)
   - `python -m src.recognize`   (tracks face X and publishes pan angles)
5. Keys in recognize: `m` MQTT on/off, `i` invert pan if the servo turns the wrong way.

MQTT: `broker.benax.rw:1883` topic `face-recognition/servo/pan` (payload = angle int). 
