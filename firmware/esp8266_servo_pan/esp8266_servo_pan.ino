#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

const char* WIFI_SSID = "Y3A";
const char* WIFI_PASS = "RCA@2024";

const char* MQTT_BROKER = "broker.benax.rw";
const int MQTT_PORT = 1883;
const char* MQTT_TOPIC = "face-recognition/servo/pan";
// Built at runtime from chip id so two boards (or a stale session) do not kick each other off.
char MQTT_CLIENT_ID[32];
const IPAddress MQTT_BROKER_IP(0, 0, 0, 0); // Optional fallback. Keep 0.0.0.0 to use MQTT_BROKER.

const int SERVO_PIN = D1; // GPIO5 on NodeMCU / ESP-12E
const int MIN_ANGLE = 20;
const int MAX_ANGLE = 160;
const int CENTER_ANGLE = 90;
const int SERVO_MIN_US = 500;
const int SERVO_MAX_US = 2400;

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
Servo panServo;
int currentAngle = CENTER_ANGLE;

void moveServoTo(int requestedAngle) {
  currentAngle = constrain(requestedAngle, MIN_ANGLE, MAX_ANGLE);
  int pulseUs = map(currentAngle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  panServo.writeMicroseconds(pulseUs);
  Serial.print("Servo angle: ");
  Serial.print(currentAngle);
  Serial.print(" pulse_us=");
  Serial.println(pulseUs);
}

void servoSelfTest() {
  Serial.print("Servo pin GPIO: ");
  Serial.println(SERVO_PIN);
  Serial.println("Servo self-test: center -> left -> right -> center");
  moveServoTo(CENTER_ANGLE);
  delay(1200);
  moveServoTo(MIN_ANGLE);
  delay(1200);
  moveServoTo(MAX_ANGLE);
  delay(1200);
  moveServoTo(CENTER_ANGLE);
  delay(1200);
}

void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  char message[16];
  unsigned int copyLength = min(length, sizeof(message) - 1);
  memcpy(message, payload, copyLength);
  message[copyLength] = '\0';

  int angle = atoi(message);
  Serial.print("MQTT ");
  Serial.print(topic);
  Serial.print(" -> ");
  Serial.println(message);
  moveServoTo(angle);
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.println();
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("WiFi connected. IP address: ");
  Serial.println(WiFi.localIP());
  Serial.print("Gateway: ");
  Serial.println(WiFi.gatewayIP());
  Serial.print("DNS: ");
  Serial.println(WiFi.dnsIP());
  Serial.print("RSSI: ");
  Serial.println(WiFi.RSSI());
}

bool brokerTcpReachable() {
  WiFiClient testClient;
  bool ok;

  if (MQTT_BROKER_IP == IPAddress(0, 0, 0, 0)) {
    IPAddress resolvedIp;
    Serial.print("Resolving MQTT broker ");
    Serial.print(MQTT_BROKER);
    Serial.print(" ... ");
    if (!WiFi.hostByName(MQTT_BROKER, resolvedIp)) {
      Serial.println("DNS failed");
      return false;
    }
    Serial.println(resolvedIp);

    Serial.print("Testing TCP ");
    Serial.print(resolvedIp);
    Serial.print(":");
    Serial.print(MQTT_PORT);
    Serial.print(" ... ");
    ok = testClient.connect(resolvedIp, MQTT_PORT);
  } else {
    Serial.print("Testing TCP ");
    Serial.print(MQTT_BROKER_IP);
    Serial.print(":");
    Serial.print(MQTT_PORT);
    Serial.print(" ... ");
    ok = testClient.connect(MQTT_BROKER_IP, MQTT_PORT);
  }

  if (ok) {
    Serial.println("ok");
    testClient.stop();
    return true;
  }

  Serial.println("failed");
  return false;
}

void connectMqtt() {
  while (!mqttClient.connected()) {
    Serial.print("Connecting to MQTT ");
    Serial.print(MQTT_BROKER);
    Serial.print(":");
    Serial.print(MQTT_PORT);
    Serial.print(" as ");
    Serial.print(MQTT_CLIENT_ID);
    Serial.print(" ... ");

    if (mqttClient.connect(MQTT_CLIENT_ID)) {
      Serial.println("connected");
      mqttClient.subscribe(MQTT_TOPIC);
      Serial.print("Subscribed to: ");
      Serial.println(MQTT_TOPIC);
      return;
    }

    Serial.print("failed, rc=");
    Serial.println(mqttClient.state());
    brokerTcpReachable(); // diagnose only after a failed connect
    Serial.println("retrying in 3 seconds");
    delay(3000);
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  snprintf(MQTT_CLIENT_ID, sizeof(MQTT_CLIENT_ID), "esp8266-pan-%08X", ESP.getChipId());

  panServo.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  servoSelfTest();

  connectWiFi();
  if (MQTT_BROKER_IP == IPAddress(0, 0, 0, 0)) {
    mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  } else {
    mqttClient.setServer(MQTT_BROKER_IP, MQTT_PORT);
  }
  mqttClient.setKeepAlive(30);
  mqttClient.setSocketTimeout(5);
  mqttClient.setCallback(onMqttMessage);
  connectMqtt();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }
  if (!mqttClient.connected()) {
    Serial.println("MQTT dropped; reconnecting...");
    delay(500);
    connectMqtt();
  }
  mqttClient.loop();
}
