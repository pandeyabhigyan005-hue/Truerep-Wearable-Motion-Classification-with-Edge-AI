#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <WiFiClientSecure.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#include "model.h"

// --- NETWORK AND MQTT CONFIGURATION ---
#include "secrets.h"
const char* MQTT_TOPIC    = "cps_wearable/bicep_twin/telemetry";
const char* MQTT_CONTROL_TOPIC = "cps_wearable/bicep_twin/control";

enum ExerciseMode : uint8_t { EXERCISE_BICEP_CURL, EXERCISE_LATERAL_RAISE };
ExerciseMode selected_exercise = EXERCISE_BICEP_CURL;
bool calibration_requested = false;

const char* exerciseName() {
  return selected_exercise == EXERCISE_LATERAL_RAISE
    ? "lateral_raise" : "bicep_curl";
}

WiFiClientSecure espClient;
PubSubClient mqttClient(espClient);

const int HAPTIC_PIN  = 18;
const int I2C_SDA_PIN = 21;
const int I2C_SCL_PIN = 22;

Adafruit_MPU6050 mpu;

unsigned long prevMicrosDSP = 0;
const unsigned long INTERVAL_DSP = 10000;  // 10 ms / 100 Hz

unsigned long prevMillisCloud = 0;
const unsigned long INTERVAL_CLOUD = 50;   // 50 ms / 20 Hz

const int WINDOW_SIZE = 50;
float imu_buffer[WINDOW_SIZE][6];
int buffer_index = 0;
bool buffer_full = false;
int inference_step = 0;

// Inference runs every 50 ms; require three bad votes and two good votes.
const int BAD_ENTRY_CONFIRMATIONS = 3;
const int GOOD_EXIT_CONFIRMATIONS = 2;
const int NEUTRAL_CONFIRMATIONS = 2;
const float MIN_FORM_VOTE_SCORE = 0.60f;
const float MIN_LATERAL_BAD_SCORE = 0.75f;
const int UNCERTAIN_BAD_RECOVERY_CONFIRMATIONS = 2;

// Neutral detection uses calibrated acceleration orientation, even during wrist motion.
const float NEUTRAL_ACCEL_TOLERANCE = 2.0f;  // m/s^2 from calibrated orientation

int bad_form_confirmations = 0;
int good_form_confirmations = 0;
int neutral_confirmations = 0;
int uncertain_form_confirmations = 0;

const float ALPHA = 0.95f;
const float DT = 0.01f;

float filtered_pitch = 0.0f;
float gyro_x_bias = 0.0f;
float gyro_y_bias = 0.0f;
float gyro_z_bias = 0.0f;
float pitch_home_offset = 0.0f;

float neutral_accel_x = 0.0f;
float neutral_accel_y = 0.0f;
float neutral_accel_z = 0.0f;

int form_state = 0;
int prev_form_state = 0;

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  if (strcmp(topic, MQTT_CONTROL_TOPIC) != 0 || length == 0) return;
  String message;
  message.reserve(length + 1);
  for (unsigned int i = 0; i < length; i++) message += static_cast<char>(payload[i]);

  const bool request_calibration =
    message.indexOf("\"calibrate\":true") >= 0;
  bool exercise_selected = false;
  if (message.indexOf("lateral_raise") >= 0) {
    selected_exercise = EXERCISE_LATERAL_RAISE;
    exercise_selected = true;
  } else if (message.indexOf("bicep_curl") >= 0) {
    selected_exercise = EXERCISE_BICEP_CURL;
    exercise_selected = true;
  } else if (!request_calibration) {
    return;
  }

  if (exercise_selected) {
    bad_form_confirmations = 0;
    good_form_confirmations = 0;
    neutral_confirmations = 0;
    uncertain_form_confirmations = 0;
    form_state = 0;
    digitalWrite(HAPTIC_PIN, LOW);
    Serial.printf("Exercise selected: %s\n", exerciseName());
  }

  if (request_calibration) {
    calibration_requested = true;
    Serial.println("Neutral recalibration requested from dashboard.");
  }
}

// Serial training capture: send N (neutral), G (good), B (bad), or X (stop).
// Captures the same six raw IMU values used by the on-device feature pipeline.
int capture_label = -1;
int i2c_fail_count = 0;

void recoverI2CBus() {
  Wire.end();

  pinMode(I2C_SDA_PIN, INPUT_PULLUP);
  pinMode(I2C_SCL_PIN, OUTPUT);

  digitalWrite(I2C_SCL_PIN, HIGH);
  delayMicroseconds(10);

  for (int i = 0; i < 9; i++) {
    digitalWrite(I2C_SCL_PIN, LOW);
    delayMicroseconds(10);
    digitalWrite(I2C_SCL_PIN, HIGH);
    delayMicroseconds(10);
  }

  pinMode(I2C_SDA_PIN, OUTPUT);
  digitalWrite(I2C_SDA_PIN, LOW);
  delayMicroseconds(10);

  digitalWrite(I2C_SCL_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(I2C_SDA_PIN, HIGH);
  delayMicroseconds(10);

  pinMode(I2C_SDA_PIN, INPUT);
  pinMode(I2C_SCL_PIN, INPUT);

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(100000);
  Wire.setTimeOut(25);
}

bool initMPU() {
  recoverI2CBus();

  if (!mpu.begin()) {
    Serial.println("MPU6050 initialization failed.");
    return false;
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  i2c_fail_count = 0;
  return true;
}

void connectNetwork() {
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.disconnect(true);
    delay(100);

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int retries = 0;
    while (WiFi.status() != WL_CONNECTED && retries < 30) {
      delay(500);
      retries++;
    }
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi connection failed.");
    return;
  }

  if (!mqttClient.connected()) {
    espClient.setInsecure();
    espClient.setTimeout(15);

    mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
    mqttClient.setKeepAlive(15);
    mqttClient.setCallback(mqttCallback);

    String clientId = "ESP32Twin-" + String(random(0xffff), HEX);

    if (mqttClient.connect(clientId.c_str(), MQTT_USER, MQTT_PASSWORD)) {
      Serial.println("Connected to HiveMQ Cloud TLS.");
      mqttClient.subscribe(MQTT_CONTROL_TOPIC, 1);
    } else {
      Serial.print("MQTT connection failed, state=");
      Serial.println(mqttClient.state());
    }
  }
}

void calibrateSensors() {
  Serial.println(
    "Calibrating: strap the sensor and hold the intended neutral position still."
  );

  float accel_x_sum = 0.0f;
  float accel_y_sum = 0.0f;
  float accel_z_sum = 0.0f;
  float gyro_x_sum = 0.0f;
  float gyro_y_sum = 0.0f;
  float gyro_z_sum = 0.0f;
  float pitch_sum = 0.0f;

  const int SAMPLES = 200;  // About 2 seconds at 100 Hz.

  for (int i = 0; i < SAMPLES; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    accel_x_sum += a.acceleration.x;
    accel_y_sum += a.acceleration.y;
    accel_z_sum += a.acceleration.z;

    gyro_x_sum += g.gyro.x * (180.0f / PI);
    gyro_y_sum += g.gyro.y * (180.0f / PI);
    gyro_z_sum += g.gyro.z * (180.0f / PI);

    pitch_sum += atan2f(a.acceleration.x, a.acceleration.z) * (180.0f / PI);

    delay(10);
  }

  neutral_accel_x = accel_x_sum / SAMPLES;
  neutral_accel_y = accel_y_sum / SAMPLES;
  neutral_accel_z = accel_z_sum / SAMPLES;

  gyro_x_bias = gyro_x_sum / SAMPLES;
  gyro_y_bias = gyro_y_sum / SAMPLES;
  gyro_z_bias = gyro_z_sum / SAMPLES;

  pitch_home_offset = pitch_sum / SAMPLES;
  filtered_pitch = 0.0f;

  Serial.println("Sensor calibration complete.");
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  delay(500);

  pinMode(HAPTIC_PIN, OUTPUT);
  digitalWrite(HAPTIC_PIN, LOW);

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(100000);

  if (!initMPU()) {
    while (true) {
      delay(10);
    }
  }

  calibrateSensors();
  connectNetwork();

  prevMicrosDSP = micros();
}

void loop() {
  while (Serial.available() > 0) {
    const char command = static_cast<char>(Serial.read());
    if (command == 'N' || command == 'n') capture_label = 0;
    else if (command == 'G' || command == 'g') capture_label = 1;
    else if (command == 'B' || command == 'b') capture_label = 2;
    else if (command == 'X' || command == 'x') capture_label = -1;
  }

  static unsigned long lastNetworkRetry = 0;

  const unsigned long nowMs = millis();

  // Retry Wi-Fi or MQTT if disconnected; service MQTT while connected.
  if (WiFi.status() != WL_CONNECTED || !mqttClient.connected()) {
    if (nowMs - lastNetworkRetry >= 3000) {
      lastNetworkRetry = nowMs;
      connectNetwork();
    }
  } else {
    mqttClient.loop();
  }

  if (calibration_requested) {
    calibration_requested = false;
    bad_form_confirmations = 0;
    good_form_confirmations = 0;
    neutral_confirmations = 0;
    uncertain_form_confirmations = 0;
    form_state = 0;
    prev_form_state = 0;
    buffer_index = 0;
    buffer_full = false;
    inference_step = 0;
    digitalWrite(HAPTIC_PIN, LOW);
    calibrateSensors();
  }

  // Read and process the IMU at 100 Hz.
  const unsigned long currentMicros = micros();

  if (currentMicros - prevMicrosDSP >= INTERVAL_DSP) {
    prevMicrosDSP += INTERVAL_DSP;

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    i2c_fail_count = 0;

    const float gx = g.gyro.x * (180.0f / PI);
    const float gy = (g.gyro.y * (180.0f / PI)) - gyro_y_bias;
    const float gz = g.gyro.z * (180.0f / PI);

    const float raw_pitch =
      (atan2f(a.acceleration.x, a.acceleration.z) * (180.0f / PI))
      - pitch_home_offset;

    filtered_pitch =
      ALPHA * (filtered_pitch + gy * DT)
      + (1.0f - ALPHA) * raw_pitch;

    // Keep the model's existing six sensor inputs unchanged.
    imu_buffer[buffer_index][0] = a.acceleration.x;
    imu_buffer[buffer_index][1] = a.acceleration.y;
    imu_buffer[buffer_index][2] = a.acceleration.z;
    imu_buffer[buffer_index][3] = gx;
    imu_buffer[buffer_index][4] = gy;
    imu_buffer[buffer_index][5] = gz;

    // Headerless CSV rows: ax,ay,az,gx,gy,gz,label. Python's training
    // pipeline can collect these rows without mixing exercise datasets.
    if (capture_label >= 0) {
      Serial.printf("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
        a.acceleration.x, a.acceleration.y, a.acceleration.z,
        g.gyro.x * (180.0f / PI),
        g.gyro.y * (180.0f / PI),
        g.gyro.z * (180.0f / PI), capture_label);
    }

    buffer_index++;

    if (buffer_index >= WINDOW_SIZE) {
      buffer_index = 0;
      buffer_full = true;
    }

    inference_step++;

    // Run model inference every 5 samples (about 50 ms) using the rolling window.
    if (buffer_full && inference_step >= 5) {
      inference_step = 0;

      double features[24];
      int f_ptr = 0;

      for (int axis = 0; axis < 6; axis++) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        float max_val = 0.0f;

        for (int i = 0; i < WINDOW_SIZE; i++) {
          const float value =
            imu_buffer[(buffer_index + i) % WINDOW_SIZE][axis];

          sum += value;
          sq_sum += value * value;

          if (fabsf(value) > max_val) {
            max_val = fabsf(value);
          }
        }

        const float mean = sum / WINDOW_SIZE;
        const float variance =
          (sq_sum / WINDOW_SIZE) - (mean * mean);
        const float std_dev = sqrtf(fmaxf(0.0f, variance));
        const float rms = sqrtf(sq_sum / WINDOW_SIZE);

        features[f_ptr++] = static_cast<double>(mean);
        features[f_ptr++] = static_cast<double>(std_dev);
        features[f_ptr++] = static_cast<double>(max_val);
        features[f_ptr++] = static_cast<double>(rms);
      }

      double class_scores[3] = {0.0, 0.0, 0.0};
#ifdef MODEL_HAS_EXERCISE_MODELS
      if (selected_exercise == EXERCISE_LATERAL_RAISE) {
        lateral_raise_score(features, class_scores);
      } else {
        bicep_curl_score(features, class_scores);
      }
#else
      // Backward-compatible with the existing binary bicep model until the
      // training pipeline generates the dual-exercise three-class model.h.
      double legacy_scores[2] = {0.0, 0.0};
      score(features, legacy_scores);
      class_scores[0] = legacy_scores[0]; // GOOD FORM
      class_scores[2] = legacy_scores[1]; // BAD FORM
#endif

#ifdef MODEL_HAS_EXERCISE_MODELS
      // The upper-arm sensor often cannot distinguish a resting bicep-curl
      // pose from the start/end of a correct curl. In bicep mode, fold the
      // learned neutral probability into GOOD FORM and expose only GOOD/BAD.
      // Lateral raises retain their separate neutral class.
      if (selected_exercise == EXERCISE_BICEP_CURL) {
        class_scores[1] += class_scores[0];
        class_scores[0] = 0.0;
      }
#endif

      // Match the recent acceleration window to the calibrated neutral pose.
      const float neutral_dx =
        static_cast<float>(features[0]) - neutral_accel_x;
      const float neutral_dy =
        static_cast<float>(features[4]) - neutral_accel_y;
      const float neutral_dz =
        static_cast<float>(features[8]) - neutral_accel_z;

      const float neutral_accel_error =
        sqrtf(
          neutral_dx * neutral_dx +
          neutral_dy * neutral_dy +
          neutral_dz * neutral_dz
        );

      // Use the calibrated pose as a neutral cue for lateral raises. Bicep
      // curl neutral probability has already been folded into GOOD FORM above.
      const bool accel_neutral_candidate =
        neutral_accel_error <= NEUTRAL_ACCEL_TOLERANCE;
#ifdef MODEL_HAS_EXERCISE_MODELS
      const bool neutral_candidate =
        selected_exercise == EXERCISE_LATERAL_RAISE &&
        accel_neutral_candidate;
#else
      const bool neutral_candidate = accel_neutral_candidate;
#endif
      if (neutral_candidate) {
        neutral_confirmations++;
        bad_form_confirmations = 0;
        good_form_confirmations = 0;
        uncertain_form_confirmations = 0;

        // Clear to good only after stable calibrated-neutral readings.
        if (neutral_confirmations >= NEUTRAL_CONFIRMATIONS) {
          form_state = 0;
        }
      } else {
        neutral_confirmations = 0;

#ifdef MODEL_HAS_EXERCISE_MODELS
        // The generated model includes an explicit resting class.
        if (class_scores[0] >= MIN_FORM_VOTE_SCORE) {
          bad_form_confirmations = 0;
          good_form_confirmations = 0;
          uncertain_form_confirmations = 0;
          form_state = 0;
        } else {
          bool ml_bad = class_scores[2] >= MIN_FORM_VOTE_SCORE;
          bool ml_good = class_scores[1] >= MIN_FORM_VOTE_SCORE;
          if (selected_exercise == EXERCISE_BICEP_CURL) {
            // Treat bicep mode as a binary decision: BAD must beat the
            // combined neutral/good score by a margin. Ambiguous windows
            // default to GOOD so small score shifts do not create false
            // bad-form alerts after neutral was merged into good.
            ml_bad = class_scores[2] >= MIN_FORM_VOTE_SCORE &&
                     class_scores[2] >= class_scores[1] + 0.10;
            ml_good = !ml_bad;
          } else {
            // Live lateral scores during false alerts reached about 0.68 BAD.
            // Require stronger model evidence before raising an alert.
            ml_bad = class_scores[2] >= MIN_LATERAL_BAD_SCORE;
          }
#else
        const bool ml_bad = class_scores[2] >= MIN_FORM_VOTE_SCORE;
        const bool ml_good = class_scores[0] >= MIN_FORM_VOTE_SCORE;
#endif

        // Ignore weak model votes to reduce brief bad-form flicker during movement.
#ifdef MODEL_HAS_EXERCISE_MODELS
        if (class_scores[0] < MIN_FORM_VOTE_SCORE) {
#endif
        if (ml_bad) {
          bad_form_confirmations++;
          good_form_confirmations = 0;
          uncertain_form_confirmations = 0;

          if (bad_form_confirmations >= BAD_ENTRY_CONFIRMATIONS) {
            form_state = 2;
          }
        } else if (ml_good) {
          good_form_confirmations++;
          bad_form_confirmations = 0;
          uncertain_form_confirmations = 0;

          if (good_form_confirmations >= GOOD_EXIT_CONFIRMATIONS) {
            form_state = 1;
          }
        } else {
          // For lateral raises, a non-neutral window without strong BAD or
          // GOOD evidence resolves to GOOD after two checks. This prevents a
          // weak, stale BAD score from remaining latched. Bicep behavior
          // retains the prior hold-on-uncertainty rule.
          bad_form_confirmations = 0;
          good_form_confirmations = 0;
          if (selected_exercise == EXERCISE_LATERAL_RAISE) {
            uncertain_form_confirmations++;
            if (uncertain_form_confirmations >= UNCERTAIN_BAD_RECOVERY_CONFIRMATIONS) {
              form_state = 1;
              uncertain_form_confirmations = 0;
            }
          } else {
            uncertain_form_confirmations = 0;
          }
        }
#ifdef MODEL_HAS_EXERCISE_MODELS
        }
#endif
      }

#ifdef MODEL_HAS_EXERCISE_MODELS
      // Temporary diagnostic: inspect whether false lateral alerts originate
      // in the model scores or in the state confirmation logic.
      if (selected_exercise == EXERCISE_LATERAL_RAISE) {
        Serial.printf(
          "LR_SCORE,N=%.3f,G=%.3f,B=%.3f,ACC_ERR=%.2f,ACC_NEUTRAL=%d,STATE=%d,BAD_VOTES=%d,GOOD_VOTES=%d\n",
          class_scores[0], class_scores[1], class_scores[2], neutral_accel_error,
          accel_neutral_candidate ? 1 : 0, form_state,
          bad_form_confirmations, good_form_confirmations
        );
      }
#endif
    }

    digitalWrite(HAPTIC_PIN, form_state == 2 ? HIGH : LOW);
  }

  // Publish telemetry every 50 ms while MQTT is connected.
  const unsigned long currentMillis = millis();

  if (currentMillis - prevMillisCloud >= INTERVAL_CLOUD) {
    prevMillisCloud = currentMillis;

    const bool state_changed = (form_state != prev_form_state);
    prev_form_state = form_state;

    if (mqttClient.connected()) {
      char payload[160];

      snprintf(
        payload,
        sizeof(payload),
        "{\"pitch\":%.2f,\"state\":%d,\"state_name\":\"%s\",\"exercise\":\"%s\",\"haptic\":%d,\"transition\":%d,\"uptime\":%lu}",
        filtered_pitch,
        form_state,
        form_state == 2 ? "BAD FORM" :
          (form_state == 1 ? "GOOD FORM" :
            (selected_exercise == EXERCISE_BICEP_CURL ? "GOOD FORM" : "NEUTRAL")),
        exerciseName(),
        form_state == 2 ? 1 : 0,
        state_changed ? 1 : 0,
        currentMillis
      );

      mqttClient.publish(MQTT_TOPIC, payload);
    }
  }
}
}