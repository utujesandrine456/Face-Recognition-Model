# src/servo_test.py
"""
Send test pan angles to the ESP8266 over MQTT (no camera needed).

Run: python -m src.servo_test
"""
from __future__ import annotations

import time

from .mqtt_servo import CENTER_ANGLE, MAX_ANGLE, MIN_ANGLE, ServoPanPublisher


def main():
    pub = ServoPanPublisher(client_id="pc-servo-test")
    pub.connect()

    sequence = [
        ("center", CENTER_ANGLE),
        ("left", MIN_ANGLE),
        ("right", MAX_ANGLE),
        ("center", CENTER_ANGLE),
    ]

    print("Publishing servo self-test sequence over MQTT...")
    try:
        for name, angle in sequence:
            print(f"-> {name} ({angle})")
            pub.publish_angle(angle, force=True)
            time.sleep(1.5)
    finally:
        pub.close()

    print("Done. Watch Serial Monitor on the ESP8266 for 'MQTT ... -> <angle>'.")


if __name__ == "__main__":
    main()
