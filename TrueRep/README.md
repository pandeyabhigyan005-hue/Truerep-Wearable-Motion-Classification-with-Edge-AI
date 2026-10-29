# TrueRep

TrueRep is a wearable edge-ML prototype for classifying sensor-observable movement patterns and providing haptic feedback. The current pilot uses separate Random Forest models for bicep curls and lateral raises. Rehabilitation monitoring is a potential future application and has not been clinically validated.

## Repository layout

```text
TrueRep/
├── firmware/
│   ├── TrueRep.ino
│   ├── model.h
│   └── secrets.example.h
├── ml/
│   ├── train_features.py
│   ├── requirements.txt
│   └── data/
│       ├── real_neutral.csv
│       ├── real_good_form.csv
│       ├── real_bad_form.csv
│       ├── lateral_raise_good_form.csv
│       └── lateral_raise_bad_form.csv
├── cloud/
│   ├── cloud_bridge.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html
│   └── assets/arm-rig.glb
└── README.md
```

## System overview

An upper-arm MPU6050 supplies three-axis acceleration and gyroscope data to an ESP32 over I2C. Firmware samples at a nominal 100 Hz, computes 24 statistics from a rolling 50-sample window, and runs the selected exercise model locally. Confirmed BAD FORM state drives the vibration output. MQTT carries telemetry to the browser dashboard and, optionally, the Firebase logging bridge. Exercise selection is provided by the dashboard; the device does not automatically identify an unknown exercise.

## Firmware setup

1. Install Arduino IDE or Arduino CLI and an ESP32 board package.
2. Install these Arduino libraries: Adafruit MPU6050, Adafruit Unified Sensor, and PubSubClient. `Wire`, `WiFi`, and `WiFiClientSecure` are provided by the ESP32 core.
3. Copy `firmware/secrets.example.h` to `firmware/secrets.h` and set Wi-Fi and MQTT values. Never commit `secrets.h`.
4. Open `firmware/TrueRep.ino`, select the appropriate ESP32 board and port, then upload. The included `model.h` contains generated exercise-specific C/C++ scoring functions.

## Train or regenerate models

```bash
cd ml
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python train_features.py
```

Training reads the included CSV recordings and writes the generated `firmware/model.h`. It reports balanced accuracy and confusion matrices. The 80% chronological training split, 50-sample gap, and remaining validation tail operate within each label-consistent segment; this is not an unseen-user evaluation.

## Dashboard

Serve `frontend/` over a local HTTP server (rather than opening the HTML with `file://`) so the 3D asset can load. Configure the MQTT WebSocket endpoint and dashboard connection fields for your broker. The dashboard screenshot provided with the project showed demo mode; configure a live broker before expecting a device stream.

## Optional Firebase alert bridge

```bash
cd cloud
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
```

Set the variables listed in `cloud/.env.example` in your shell, and point `GOOGLE_APPLICATION_CREDENTIALS` to a Firebase service-account JSON file stored outside this repository. Then run `python cloud_bridge.py`. Check Firebase directly to verify saved events.

## Scope and limitations

The single upper-arm IMU measures that sensor's motion; it does not directly measure elbow angle, muscle activation, or whole-body posture. Results depend on placement, calibration, training examples, and consistent preprocessing. The cloud bridge is optional; model inference runs on the ESP32. This is a prototype, not a medical device or clinically validated rehabilitation system.

## Hardware pins

- MPU6050 SDA: GPIO 21
- MPU6050 SCL: GPIO 22
- Haptic control: GPIO 18 (use a suitable motor driver; do not power a motor directly from a GPIO)

## License

Add a license before making this repository public. Confirm licenses for the included dashboard assets and third-party dependencies before redistribution.
