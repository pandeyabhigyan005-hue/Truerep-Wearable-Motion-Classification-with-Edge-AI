"""Optional MQTT-to-Firebase event bridge for TrueRep."""
import json
import os
import time
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, db
import paho.mqtt.client as mqtt

BROKER=os.environ["MQTT_BROKER"]
PORT=int(os.getenv("MQTT_PORT", "8883"))
USERNAME=os.environ["MQTT_USERNAME"]
PASSWORD=os.environ["MQTT_PASSWORD"]
TOPIC=os.getenv("MQTT_TOPIC", "cps_wearable/bicep_twin/telemetry")
DATABASE_URL=os.environ["FIREBASE_DATABASE_URL"]
last_state=None

def log_event(kind, data):
    payload={"timestamp_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "epoch_time":time.time(),"event":kind,"state":data.get("state"),
             "state_name":data.get("state_name"),"pitch":data.get("pitch"),
             "exercise":data.get("exercise"),"uptime":data.get("uptime")}
    db.reference("exercise_logs").push(payload)

def on_message(client, userdata, msg):
    global last_state
    try:
        data=json.loads(msg.payload.decode("utf-8"))
        state=data.get("state")
        if state != last_state:
            if state == 2: log_event("BAD_FORM", data)
            elif state == 1 and last_state == 2: log_event("RECOVERED", data)
            last_state=state
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        print(f"Skipping invalid telemetry: {exc}")

def main():
    cred_path=os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    firebase_admin.initialize_app(credentials.Certificate(cred_path), {"databaseURL":DATABASE_URL})
    client=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()
    client.on_message=on_message
    client.connect(BROKER,PORT,keepalive=60)
    client.subscribe(TOPIC)
    print("TrueRep cloud bridge listening for telemetry")
    client.loop_forever()

if __name__ == "__main__": main()
