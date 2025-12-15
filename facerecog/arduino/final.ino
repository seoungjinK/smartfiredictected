#include <LiquidCrystal_I2C.h>
#include <Arduino.h>
#include <Servo.h> // 서보 모터 제어를 위한 라이브러리


// 핀 설정
#define GAS_SENSOR_PIN A0       // 가스 센서 핀
#define TEMP_SENSOR_PIN A1      // 온도 센서 핀
#define BUZZER_PIN 12           // 부저 핀
#define SERVO_PIN 10            // 서보 모터 핀
#define LED1_PIN 2              // LED1 핀
#define LED2_PIN 3              // LED2 핀
#define LED3_PIN 4              // LED3 핀
#define LED4_PIN 5              // LED4 핀
#define LED5_PIN 6              // LED5 핀
#define soundSensorPin 8        // 사운드 센서 핀


LiquidCrystal_I2C lcd(0x27, 16, 2); // LCD 주소 0x3F, 크기 16x2
// 상수 정의
#define GAS_THRESHOLD 400       // 가스 감지 임계값
#define TEMP_THRESHOLD 50.0     // 온도 임계값 (°C)
#define ADC_MAX 1023.0          // ADC 최대값 (10비트)


// 전역 변수
float temperature = 0.0;
int gasValue = 0;
bool fireState = false;
bool fireStates = false;
bool humanState = false;
bool humanDetected = false;
bool manualControl = false; // 수동 제어 상태
int soundDetected = LOW;    // 사운드 감지 상태 저장 변수
unsigned long previousMillis = 0; // 데이터 전송 간격 제어
unsigned long servoActionMillis = 0; // 서보 모터 상태 변경 시간 기록
const unsigned long dataInterval = 1000; // 센서 데이터 전송 간격 1초
const unsigned long servoDelay = 10000;  // 서보 모터 상태 변경 대기 시간 10초

bool buzzerState = false;
unsigned long buzzerStartMillis = 0;
const unsigned long buzzerDuration = 500;

// 서보 모터 객체 생성
Servo servo;

// 부저 제어 함수
void buzzerOn(int frequency) {
  if(!buzzerState){
    tone(BUZZER_PIN, frequency);
    buzzerState = true;
    buzzerStartMillis = millis();
  }
}

void buzzerOff() {
  if (buzzerState){
    noTone(BUZZER_PIN);
    buzzerState = false;
  }
}

// LED 제어 함수
void ledOnSequence() {
  for (int i = 0; i < 5; i++) {
    digitalWrite(LED1_PIN + i, HIGH);
    delay(500);
  }
}

void ledOff() {
  for (int i = 0; i < 5; i++) {
    digitalWrite(LED1_PIN + i, LOW);
  }
}

// 서보 모터 각도 제어 함수
void setServoAngle(int angle) {
  servo.write(angle);
}

// TMP36 온도 센서 값 읽기
float readTemperature() {
  int tempRaw = analogRead(TEMP_SENSOR_PIN);
  float voltage = tempRaw * (5.0 / ADC_MAX); // 전압 계산
  return (voltage * 3) * 50 ; // 섭씨로 변환
}

// 센서 데이터 전송 함수
void sendSensorData() {
  Serial.print("gas: ");
  Serial.print(gasValue);
  Serial.print(", temperature: ");
  Serial.println((int)temperature); // 소수점 제거를 위해 정수 변환
}

// 시리얼 데이터 처리 함수
void processReceivedData(char data) {
  if (data == '1') {
    humanState = true;
  } else if (data == '2') {
    humanState = false;
  } else if (data == '3') {
    fireState = true;
  } else if (data == '4') {
    fireState = false;
  } else if (data == '+') {
    manualControl = true;
    setServoAngle(0); // 방화문 열기
    servoActionMillis = millis(); // 동작 시간 기록
  } else if (data == '-') {
    manualControl = true;
    setServoAngle(90); // 방화문 닫기
    servoActionMillis = millis(); // 동작 시간 기록
    delay(50);
  } else if (data == '8'){
    buzzerOn(1000);
    delay(100);
    buzzerOff();
  } else if (data == '9'){
    buzzerOff();
  }
}



void setup() {
  // 핀 모드 설정
  pinMode(GAS_SENSOR_PIN, INPUT);
  pinMode(TEMP_SENSOR_PIN, INPUT);
  pinMode(soundSensorPin, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  for (int i = 0; i < 5; i++) {
    pinMode(LED1_PIN + i, OUTPUT);
  }

  // 서보 모터 초기화
  servo.attach(SERVO_PIN);
  setServoAngle(90); // 초기 위치 설정

  // LCD 초기화
  lcd.init();          // LCD 초기화
  lcd.backlight();     // LCD 백라이트 켜기
  lcd.clear();         // LCD 화면 지우기
  lcd.setCursor(0, 0); // 커서를 첫 번째 줄로 이동
  lcd.print("System Ready");

  // 시리얼 통신 시작
  Serial.begin(9600);
}

void loop() {
  // 현재 시간 가져오기
  unsigned long currentMillis = millis();

  // 주기적으로 센서 데이터 읽기
  if (currentMillis - previousMillis >= dataInterval) {
    previousMillis = currentMillis;

    // 센서 값 읽기
    gasValue = analogRead(GAS_SENSOR_PIN);
    temperature = readTemperature();
    soundDetected = digitalRead(soundSensorPin); // 사운드 감지 상태 읽기

    // 센서 데이터 전송
    sendSensorData();
    

    // 화재 상태 업데이트
    fireStates = ((gasValue > GAS_THRESHOLD || temperature > TEMP_THRESHOLD) && fireState);
  }

  // 서보 모터 자동 제어 (수동 입력 없을 때만)
  if (!manualControl && (currentMillis - servoActionMillis >= servoDelay)) {
    // 사람이 말하는 소리가 감지되면 방화문 열기
    if (!humanState && (soundDetected == HIGH) && fireStates) {
      setServoAngle(0); // 방화문 열기
      ledOnSequence();
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Human Voice O");
      lcd.setCursor(0, 1);
      lcd.print("Fire O");
    } else if (!humanState && fireStates) {
      setServoAngle(90); // 방화문 닫기
      buzzerOn(1000);
      delay(500);
      buzzerOff();
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Human X");
      lcd.setCursor(0, 1);
      lcd.print("Fire O");
    } else if (humanState && fireStates) {
      setServoAngle(0); // 방화문 열기
      buzzerOn(1000);
      delay(500);
      ledOnSequence();
      buzzerOff();
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Human O");
      lcd.setCursor(0, 1);
      lcd.print("Fire O");
    } else if (!humanState && !fireStates) {
      buzzerOff();
      ledOff();
       lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Human X");
      lcd.setCursor(0, 1);
      lcd.print("Fire X");
    } else if (humanState && !fireStates) {
      setServoAngle(0); // 방화문 열기
      buzzerOff();
      ledOff();
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Human O");
      lcd.setCursor(0, 1);
      lcd.print("Fire X");
    }
    servoActionMillis = currentMillis; // 마지막 동작 시간 기록
  }


  // 시리얼 데이터 수신 처리
  if (Serial.available()) {
    char receivedChar = Serial.read();
    processReceivedData(receivedChar);
    manualControl = false; // 수동 입력 처리 후 자동 모드로 복귀
  }

  delay(50); // 주기적으로 센서 값 읽기
}

