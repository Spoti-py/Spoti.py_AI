# Spoti.py_AI — AI Server

간단한 소개
- 이 디렉터리는 Spoti.py_AI의 웹 서비스 컴포넌트입니다. FastAPI로 구현된 경량 마이크로서비스로, 비디오 프레임 스트리밍과 포즈(키포인트) 분석을 제공하여 모델 추론 파이프라인의 엔드포인트 역할을 합니다.

핵심 포인트
- 실시간 스트리밍: `/stream`에서 프레임을 multipart 스트림으로 전송합니다.
- 키포인트 처리: `/upload`(HTTP)와 `/ws/keypoints`(WebSocket)를 통해 키포인트 데이터를 받아 분석 결과를 반환합니다.
- 단일 진입점: `main.py`가 FastAPI 앱을 노출합니다 — 배포 및 통합이 쉽습니다.

빠른 소개용 실행 예시
- 개발 환경에서 빠르게 띄우려면:

```bash
cd web
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

구성 및 위치
- `main.py`: FastAPI 엔트리포인트 (엔드포인트 정의 포함)
- `module/`: 프레임 전처리 및 스트리밍 로직 (`frameprocesser.py`, `streaming.py` 등)
- `models/bestM.pt`: 학습된 모델 파일(로컬에서 로드됨)
- `requirements.txt`: 필요한 Python 패키지 목록

간단한 사용 시나리오
- 웹캠이나 비디오 소스에서 프레임을 스트리밍하고, 추출한 키포인트를 `/upload`로 제출하면 처리 결과(JSON)를 받습니다. 실시간 인터랙션이 필요하면 `/ws/keypoints`로 WebSocket 연결 후 JSON을 주고받으세요.
- 눈 감김 상태가 감지되면 응답에 `"alarm": true`가 포함됩니다. 프론트에서는 이 값을 보고 경고음을 켜고, `"alarm": false`가 오면 끄면 됩니다.

주의 및 참고사항
- 의존성(OpenCV, MediaPipe 등)은 `requirements.txt`를 확인하세요.
- MongoDB 연동 코드는 현재 주석 처리되어 있습니다. 필요 시 환경변수를 설정하고 관련 코드를 활성화하세요.
