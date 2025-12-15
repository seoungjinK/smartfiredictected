from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import numpy as np
import time
import serial
import threading
import atexit
import mysql.connector
import json
import torch  # YOLOv5 모델을 사용하기 위한 라이브러리
import dlib
import os

app = Flask(__name__) # flask 애플리케이션 인스턴스 생성

app.secret_key = 'your_secret_key'  # 세션 관리를 위한 시크릿 키 설정

# 사용자 얼굴을 저장할 경로 설정
recognition_person_detection_time = 0
FACE_DIR = "faces/"
os.makedirs(FACE_DIR, exist_ok=True)
last_message = ""

# dlib의 얼굴 인식기와 얼굴 랜드마크 탐지기 로드
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# 사용자 감지 상태 저장용 전역 변수
user_detection_status = {}



# 관리자 코드 설정
SUPER_ADMIN_CODE = 'super'
ADMIN_CODE = 'admin'

# 데이터 베이스 연결 설정
db_config = {
    'user': 'root',
    'password': '1234',
    'host': '127.0.0.1',
    'database': 'fire'
}
# MySQL 연결
mydb = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="1234",
    database="fire"
)


# 데이터베이스 연결을 반환하는 함수 
def get_db_connection():
    return mysql.connector.connect(**db_config)

# 기본 센서 데이터
sensor_data = {
    'gas': 0,  # 초기 값 0
    'temperature': 0  # 초기 값 0
}

# 불꽃 감지 상태를 저장할 변수
flame_detected = False 
flame_detected_temp = False
person_detected = False  # 사람 감지 상태 변수

# 불꽃 감지 시간을 기록하는 변수
flame_detection_time = 0  # 불꽃 감지 시작 시간 (초 단위)
flame_detection_time1 = 0

# 사람 감지 시간을 기록하는 변수
person_detection_time = 0  # 사람 감지 시작 시간

# 비디오 캡처 객체 생성
cap = None
lock = threading.Lock()


# 배경 제거 객체 생성
fgbg = cv2.createBackgroundSubtractorMOG2()

# YOLOv5 모델 로드 (최신 버전의 YOLOv5 모델)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s'는 경량 모델, 'yolov5m', 'yolov5l', 'yolov5x' 등도 가능

# 불꽃의 HSV 색상 범위 정의 (좀 더 세밀한 범위 설정)
lower_yellow = np.array([20, 150, 150])  # 노란색 (Hue 20~30), 채도 150 이상, 명도 150 이상
upper_yellow = np.array([30, 255, 255])  # 노란색 (Hue 20~30), 채도 255, 명도 255

lower_orange = np.array([10, 150, 150])  # 주황색 (Hue 10~20), 채도 150 이상, 명도 150 이상
upper_orange = np.array([20, 255, 255])  # 주황색 (Hue 10~20), 채도 255, 명도 255

lower_red1 = np.array([0, 150, 150])  # 빨간색 1 (Hue 0~10), 채도 150 이상, 명도 150 이상
upper_red1 = np.array([10, 255, 255])  # 빨간색 1 (Hue 0~10), 채도 255, 명도 255

lower_red2 = np.array([160, 150, 150])  # 빨간색 2 (Hue 160~180), 채도 150 이상, 명도 150 이상
upper_red2 = np.array([180, 255, 255])  # 빨간색 2 (Hue 160~180), 채도 255, 명도 255

# 시리얼 포트 설정
def initialize_serial(port='COM8', baudrate=9600, timeout=1):
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)   # 시리얼 포트 초기화 대기
        print(f"{port}에 성공적으로 연결되었습니다.")
        return ser
    except serial.SerialException as e:
        print(f"시리얼 포트를 열 수 없습니다: {e}")
        return None


# 시리얼 통신으로 상태 전송 함수
def send_serial_message(message):
    if ser and ser.is_open:
        ser.write(message.encode()) # 메세지를 바이트로 인코딩하여 전송
        print(f"Serial message sent: {message}")
    else:
        print("Serial port is not open.")


# 시리얼 통신을 별도의 스레드에서 처리하는 함수
def serial_communication():
    global ser,sensor_data
    while ser:
        try:
            if ser.in_waiting > 0:  # 버퍼에 읽을 데이터가 있으면
                data = ser.readline().decode('utf-8').strip()
                # 데이터 형식: "gas: 45, temperature: 22"
                if data.startswith('gas') and 'temperature' in data:
                    # 예시 데이터 처리
                    parts = data.split(',')
                    gas_level = int(parts[0].split(':')[1].strip())
                    temperature = int(parts[1].split(':')[1].strip())
                    sensor_data['gas'] = gas_level
                    sensor_data['temperature'] = temperature
        
        except serial.SerialException as e:
            print(f"시리얼 통신 중 오류: {e}")  # 시리얼 통신 오류 처리
            break  # 오류 발생 시 루프 종료


# 사용자 얼굴 등록 스트리밍 함수
# 선택한 사용자 이름을 OpenCV 카메라에 표시하는 함수
def generate_camera_stream(usernames):
    global user_detection_status  # 전역 변수 사용
    cap = cv2.VideoCapture(0)

    # 사용자 얼굴 로드
    known_faces = {}
    for username in usernames:
        face_file_path = os.path.join(FACE_DIR, f"{username}.npy")
        if os.path.exists(face_file_path):
            face_descriptor = np.load(face_file_path)
            known_faces[username] = face_descriptor

    # 사용자 감지 시간 초기화
    last_detection_time = {username: time.time() for username in usernames}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        detected_usernames = set()

        for face in faces:
            x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
            shape = shape_predictor(gray, face)
            face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)
            face_descriptor = np.array(face_descriptor, dtype=np.float32)

            # 등록된 사용자와 비교
            min_dist = float("inf")
            name = "unknown"
            for user_name, known_descriptor in known_faces.items():
                dist = np.linalg.norm(known_descriptor - face_descriptor)
                if dist < min_dist:
                    min_dist = dist
                    if dist < 0.4:
                        name = user_name

            # 결과 화면에 표시
            color = (0, 0, 255) if name == "unknown" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if name != "unknown":
                detected_usernames.add(name)
                last_detection_time[name] = time.time()

        # 감지 상태 업데이트
        current_time = time.time()
        for username in usernames:
            if username in detected_usernames:
                user_detection_status[username] = True
                send_serial_message("9")  # 얼굴이 감지되면 '9' 메시지 전송
            else:
                user_detection_status[username] = False
                if current_time - last_detection_time[username] > 10:  # 10초 동안 감지되지 않으면
                    send_serial_message("8")  # '8' 메시지 전송

        # JPEG 형식으로 인코딩하여 스트리밍
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



# 실시간 비디오 스트리밍 함수 (불꽃 및 사람 감지 포함)
def generate_video_stream():
    global flame_detected, flame_detected_temp, person_detected, flame_detection_time, flame_detection_time1, person_detection_time, cap
    cap = cv2.VideoCapture(0)
    while True:
        with lock:
            ret, img_frame = cap.read()
        if not ret:
            break
        
        # 배경 제거와 불꽃 검출
        fgmask = fgbg.apply(img_frame)  # 마스크 생성
        img_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)  # HSV 색 공간으로 변환
        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)  # 노란색 마스크
        mask_orange = cv2.inRange(img_hsv, lower_orange, upper_orange)  #주황색 마스크
        mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)    # 빨간색1 마스크
        mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)    # 빨간색2 마스크

        # 모든 불꽃 색상을 하나의 마스크로 결합
        mask_flame = cv2.bitwise_or(mask_red1, mask_red2)
        mask_flame = cv2.bitwise_or(mask_flame, mask_orange)
        mask_flame = cv2.bitwise_or(mask_flame, mask_yellow)
        mask_combined = cv2.bitwise_and(mask_flame, fgmask) # 배경 제거 마스크와 결합

        # 노이즈 제거를 위한 형태학적 연삭
        kernel = np.ones((5, 5), np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel) # 닫힘 영상
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel) # 열림 영상

        # 연결된 컴포넌트 찾기
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_combined)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 500:  # 면적 기준으로 필터링
                # 불꽃 영역에 빨간색 사각형 그리기
                cv2.rectangle(img_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                center_x, center_y = int(centroids[i][0]), int(centroids[i][1])
                # 불꽃 중신에 파란색 원 그리기
                cv2.circle(img_frame, (center_x, center_y), 5, (255, 0, 0), 3)
                # 불꽃 감지 여부 확인
                flame_detected = True

        
        # 불꽃 감지 상태 업데이트 및 시리얼 전송
        if flame_detected:
            if flame_detection_time == 0:
                flame_detection_time = time.time()  # 불꽃 감지 시간 기록
            elif time.time() - flame_detection_time >= 8:  # 8초 이상 감지되면 불꽃 감지로 간주
                flame_detected_temp = True
                send_serial_message("3")  # 불꽃 감지 메시지 전송
                flame_detected = False  # 불꽃 감지 초기화
        else:
            flame_detected_temp = False
            if flame_detection_time1 == 0:
                flame_detection_time1 = time.time()  # 불꽃 감지 해제 시간 기록
            elif time.time() - flame_detection_time1 >= 1:  # 1초 이상 불꽃 감지되지 않으면
                flame_detection_time = 0  # 불꽃 감지 상태 초기화
                send_serial_message("4")  # 불꽃 감지되지 않음 메시지 전송

        # 사람 감지
        results = model(img_frame)  # YOLOv5 모델에 입력 이미지 전달
        detections = results.xyxy[0].cpu().numpy()  # detections는 (x1, y1, x2, y2, confidence, class) 형식

        # 사람 클래스 (person)의 인덱스는 0번입니다.
        # 감지된 객체 중 사람 클래스 (class_id)만 필터링
        person_currently_detected = False  # 현재 프레임에서 사람 감지 상태 초기화
        for *box, conf, cls in detections:
            if int(cls) == 0:  # 클래스 0은 사람(class_id 0)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 사람에 녹색 경계 상자
                person_currently_detected = True
                person_detection_time = time.time()  # 사람 감지 시간 기록

        # 사람 감지 상태에 따른 시리얼 메시지 전송
        if person_currently_detected:
            # 사람 감지가 유지되는 동안 person_detected 상태 업데이트
            person_detected = True
            send_serial_message("1")  # 사람 감지 메시지 전송
        else:
            # 사람이 감지된 후 일정 시간(예: 3초) 후에만 2 신호 전송
            if person_detected and time.time() - person_detection_time > 3:
                person_detected = False
                send_serial_message("2")  # 사람 감지되지 않음 메시지 전송

        # 프레임을 JPEG 형식으로 인코딩
        ret, jpeg = cv2.imencode('.jpg', img_frame)
        if not ret:
            continue
        # JPEG 이미지를 Flask에 전달할 수 있도록 인코딩한 바이트 스트림 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


def generate_video_capture_stream(username):
    cap = cv2.VideoCapture(0)
    print(f"{username}의 얼굴을 등록하고 있습니다. 잠시만 기다려 주세요...")

    face_descriptors = []
    count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            # 등록 진행 상태를 화면에 표시
            if count < 100:
                cv2.putText(frame, f"{count}/100", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for face in faces:
                x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if len(faces) == 1 and count < 100:
                    shape = shape_predictor(gray, face)
                    face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)
                    face_descriptors.append(face_descriptor)
                    count += 1
                    print(f"{count}번째 얼굴 데이터를 저장했습니다.")

                    if count >= 100:
                        mean_descriptor = np.mean(face_descriptors, axis=0, dtype=np.float32)  # 정밀도를 높이기 위해 float32 사용
                        face_file_path = os.path.join(FACE_DIR, f"{username}.npy")
                        if os.path.exists(face_file_path):
                            existing_descriptor = np.load(face_file_path)
                            mean_descriptor = np.mean([existing_descriptor, mean_descriptor], axis=0, dtype=np.float32)
                        np.save(face_file_path, mean_descriptor)
                        print(f"{username}의 얼굴 등록이 완료되었습니다.")

            # 프레임을 JPEG 형식으로 인코딩하여 스트리밍
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()
        cv2.destroyAllWindows()


# CCTV 비디오 스트리밍 함수
def generate_video_stream_cctv():
    global cap
    cap = cv2.VideoCapture(0)
    while True:
        with lock:
            ret, img_frame = cap.read()
        if not ret:
            break
        # 프레임을 JPEG 형식으로 인코딩
        ret, jpeg = cv2.imencode('.jpg', img_frame)
        if not ret:
             continue    # 인코딩 실패 시 다음 프레임으로 넘어감
        # JPEG 이미지를 Flask에 전달할 수 있도록 인코딩한 바이트 스트림 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# 메인 페이지 라우트
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # 로그인 처리, 로그인 폼에서 사용자명과 비밀번호 추출
        username = request.form['username']
        password = request.form['password']
        # MySQL 데이터베이스에서 사용자 정보 확인
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        # 사용자명과 비밀번호로 사용자 검색
        query = "SELECT * FROM users WHERE username = %s AND password = %s"
        cursor.execute(query, (username, password))
        user = cursor.fetchone()
        # 데이터베이스 연결 종료
        cursor.close()
        conn.close()

        # 사용자가 존재하면 세션 설정 및 역할에 따라 페이지 이동
        if user:
            # print(f"Role of the user: {user['role']}")  # role 값 확인용 출력
            session['logged_in'] = True
            session['username'] = user['username']
            session['role'] = user['role']
            
            # 역할에 따라 각 페이지로 리다이렉트
            if user['role'] == 'super_admin' or user['role'] == 'admin':
                return render_template('master.html')
            else:
                return render_template('user.html')
        else:
            error = '아이디 또는 비밀번호가 잘못되었습니다.'
            return render_template('main.html', error=error)
    # GET 요청 시 로그인 페이지 렌더링
    return render_template('main.html')

# 회원가입 페이지 라우트
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'] # 회원가입 폼에서 사용자 정보 추출
        password = request.form['password']
        email = request.form['email']
        code = request.form['code'].strip()  # 코드 값 공백 제거

        # 코드 값이 비어 있으면 기본 역할을 user로 설정
        if code == '':
            role = 'user'
        elif code == 'super':
            role = 'super_admin'
        elif code == 'admin':
            role = 'admin'
        else:
            # 유효하지 않은 코드인 경우
            error = '존재하지 않는 코드입니다. 다시 입력해주세요.'
            return render_template('register.html', error=error)
        
        # 데이터베이스 연결
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 동일한 사용자명 확인
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            error = '이미 존재하는 아이디입니다. 다른 아이디를 사용해주세요.'
            cursor.close()
            conn.close()
            return render_template('register.html', error=error)
        
        # 동일한 이메일 확인
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_email = cursor.fetchone()
        if existing_email:
            error = '이미 사용 중인 이메일입니다. 다른 이메일을 사용해주세요.'
            cursor.close()
            conn.close()
            return render_template('register.html', error=error)
            
        # 새로운 사용자 추가
        cursor.execute(
            "INSERT INTO users (username, password, email, code, role) VALUES (%s, %s, %s, %s, %s)",
            (username, password, email, code, role)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return redirect(url_for('main'))

    return render_template('register.html')

# 사용자 관리 페이지 라우트
@app.route('/manage_users', methods=['GET', 'POST'])
def manage_users():
    # 'super_admin' 역할을 가진 사용자만 접근 가능
    if 'logged_in' not in session or session.get('role') != 'super_admin':
        return redirect(url_for('main'))

    if request.method == 'POST':
        # AJAX 요청으로 권한 변경 처리
        data = request.get_json()
        target_username = data.get('username')
        action = data.get('action')  # 'grant_admin', 'revoke_admin', 'grant_super_admin', 'revoke_super_admin'

        if not target_username or not action:
            return jsonify({'status': 'error', 'message': '잘못된 요청입니다.'}), 400

        # MySQL 데이터베이스 연결
        conn = get_db_connection()
        cursor = conn.cursor()

        # 사용자 존재 여부 확인
        cursor.execute("SELECT * FROM users WHERE username = %s", (target_username,))
        user = cursor.fetchone()
        if not user:
            cursor.close()
            conn.close()
            return jsonify({'status': 'error', 'message': '존재하지 않는 사용자입니다.'}), 400

        # 역할 변경 로직
        try:
            if action == 'grant_admin':
                cursor.execute("UPDATE users SET role = 'admin' WHERE username = %s", (target_username,))
            elif action == 'revoke_admin':
                cursor.execute("UPDATE users SET role = 'user' WHERE username = %s", (target_username,))
            elif action == 'grant_super_admin':
                cursor.execute("UPDATE users SET role = 'super_admin' WHERE username = %s", (target_username,))
            elif action == 'revoke_super_admin':
                cursor.execute("UPDATE users SET role = 'admin' WHERE username = %s", (target_username,))
            else:
                cursor.close()
                conn.close()
                return jsonify({'status': 'error', 'message': '알 수 없는 동작입니다.'}), 400

            conn.commit()   # 변경 사항 커밋
            cursor.close()
            conn.close()
            return jsonify({'status': 'success'})
        except Exception as e:
            conn.rollback() # 오류 발생시 롤백
            cursor.close()
            conn.close()
            return jsonify({'status': 'error', 'message': '권한 변경 중 오류가 발생했습니다.'}), 500

    # GET 요청 시 사용자 목록 조회
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT username, email, role FROM users")
    users = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('manage_users.html', users=users)

# 관리자 페이지 라우트
@app.route('/master')
def master_page():
    if 'logged_in' in session and session.get('role') == 'super_admin':
         return render_template('master.html')
    elif 'logged_in' in session and session.get('role') == 'admin':
        return render_template('master.html')
    return redirect(url_for('main'))

# 사용자 페이지 라우트
@app.route('/user')
def user_page():
    if 'logged_in' in session and session.get('role') == 'user':
        return render_template('user.html')
    return redirect(url_for('main'))

@app.route('/missing_person')
def missing_person_page():
    return render_template('missing_person.html')

# 로그아웃 라우트
@app.route('/logout')
def logout():
    # 세션 종료 등 로그아웃 처리
    session.clear()
    return redirect(url_for('main'))

# 방화문 제어 라우트
@app.route('/control_fire_door', methods=['POST'])
def control_fire_door():
    data = request.get_json()   # json에 데이터 파싱
    action = data.get('action') # 수행할 액션 ('open', 'close')

    if action == 'open':
        # 방화문 열기: 아두이노에 '+' 메시지 전송
        send_serial_message("+")
        status = 'opened'
    elif action == 'close':
        # 방화문 닫기: 아두이노에 '-' 메시지 전송
        send_serial_message("-")
        status = 'closed'
    else:
        # 알 수 없는 액션인 경우 에러 응답
        return jsonify({'status': 'unknown action'}), 400

    return jsonify({'status': status})  # 성공 응답

# 센서 데이터를 반환하는 라우트
@app.route('/get_sensor_data', methods=['GET'])
def get_sensor_data():
    global sensor_data
    return jsonify(sensor_data)

# 감지된 비디오 스트림 (불꽃 및 사람 감지)
@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# CCTV 비디오 스트림 ( 원본 비디오 )
@app.route('/video_feed_cctv')
def video_feed_cctv():
    return Response(generate_video_stream_cctv(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train_stream')
def train_stream():
    if 'logged_in' not in session or not session.get('username'):
        return "로그인된 사용자가 없습니다.", 401
    username = session.get('username')
    return Response(generate_video_capture_stream(username), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    cursor = mydb.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    cursor.close()

    # 초기 사용자 상태 설정
    global user_detection_status
    user_detection_status = {user['username']: False for user in users}
    return render_template('capture.html', users=users)

@app.route('/show_camera', methods=['GET'])
def show_camera():
    usernames = request.args.getlist('usernames')
    return Response(generate_camera_stream(usernames), mimetype='multipart/x-mixed-replace; boundary=frame')

# 사용자 감지 상태를 클라이언트에 제공
@app.route('/user_detection_status', methods=['GET'])
def get_user_detection_status():
    return jsonify(user_detection_status)

# 불꽃 감지 상태 확인
@app.route('/flame_status')
def flame_status():
    global flame_detected_temp
    return jsonify({'flame_detected': flame_detected_temp})

# 사람 감지 상태 확인
@app.route('/person_status')
def person_status():
    global person_detected
    return jsonify({'person_detected': person_detected})

# 앱 종료 시 리소스 정리
def cleanup():
    if cap.isOpened():
        cap.release()  # 비디오 캡처 객체 해제
    if ser and ser.is_open:
        ser.close()  # 시리얼 포트 닫기
    print("모든 리소스가 해제되었습니다.")

atexit.register(cleanup)

if __name__ == '__main__':
    ser = initialize_serial('COM8', 9600)
    # 시리얼 통신 스레드 시작
    if ser:
        serial_thread = threading.Thread(target=serial_communication)
        serial_thread.daemon = True # 데몬 스레드로 설정하여 메인 스레드 종료 시 자동 종료
        serial_thread.start()
    # Flask 애플리케이션 실행
    app.run(debug=True, threaded=True, use_reloader=False)
    # **옵션 설명:**
    # - debug=True: 디버그 모드 활성화 (개발 시 유용)
    # - threaded=True: 여러 요청을 동시에 처리할 수 있도록 스레딩 활성화
    # - use_reloader=False: 코드 변경 시 자동 재로딩 비활성화 (시리얼 통신과의 충돌 방지)