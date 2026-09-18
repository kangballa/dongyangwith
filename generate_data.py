import os
import json
import time
import logging
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, InternalServerError

# ---------------------------------------------------------
# 1. 보안 로깅 설정 (민감 정보 제외, 시스템 상태만 로깅)
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 2. 환경변수 기반 API Key 로드 (보안 강화 - 하드코딩 금지)
# ---------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("환경 변수에 GEMINI_API_KEY가 설정되지 않았습니다.")
    raise ValueError("보안 경고: API 인증 키 누락. 시스템을 중단합니다.")

genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------
# 3. 모델 설정 (JSON 스키마 강제로 할루시네이션 및 포맷 오류 방지)
# ---------------------------------------------------------
# gemini-1.5-flash는 대량의 정형 데이터 생성(빠른 속도, 낮은 비용)에 적합합니다.
generation_config = genai.GenerationConfig(
    response_mime_type="application/json",
    temperature=0.7, # 약간의 다양성을 위해 0.7 설정
)
model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# ---------------------------------------------------------
# 4. 프롬프트 정의 (기존 난수 기반 상관관계를 LLM 지시문으로 변환)
# ---------------------------------------------------------
def get_prompt(batch_size):
    return f"""
    당신은 보험사의 데이터 생성 AI입니다. 다음의 데이터 상관관계 규칙을 엄격히 분석하여 {batch_size}명의 가상 고객 데이터를 생성하세요.
    결과는 반드시 JSON 배열(Array of Objects) 형태로만 출력해야 합니다.

    [고객 인적사항 규칙]
    - 나이: 20~65세
    - 성별: 남성, 여성 (5:5 비율)
    - 연소득_만원: 3000~12000 사이
    - 직업위험등급: 1, 2, 3 (1등급이 가장 많음)
    - 결혼 및 자녀: 30세 미만은 대부분 미혼(자녀0), 40대 이상은 기혼 비율이 높고 자녀가 1~3명 있음.
    
    [건강 데이터 규칙]
    - 만성질환: 나이가 많을수록 고혈압, 당뇨 발병률 증가
    - 흡연여부: 비흡연 75%, 흡연 25%
    - 가족력: 없음, 암, 심혈관 중 택 1

    [보험 주계약 및 특약 규칙 (핵심)]
    1. 수호천사 암/건강보험: 암/심혈관 가족력이 있거나 흡연자일 경우 주로 가입. 
       - 20~30대 남성은 '수술비 보장 특약' 선호
       - 50~60대 여성이나 암 가족력이 있으면 '표적항암약물허가치료 특약' 압도적 선호
    2. 수호천사 간편심사(유병자)보험: 만성질환(고혈압, 당뇨)이 있는 경우 가입. 
       - 특약은 대부분 '중증질환 산정특례 보장 특약'
    3. 수호천사 우리가족 종신보험: 자녀가 있는 30~50대 기혼자가 주로 가입.
       - 특약은 '가족 수입 보장 특약'을 주로 선택
    4. 수호천사 행복 연금보험: 고소득 미혼자나 50대 이상 무자녀 가입률 높음.
       - 특약은 '특약 없음'

    [필수 JSON 키]
    나이, 성별, 연소득_만원, 직업위험등급, 결혼여부, 자녀수, 흡연여부, 만성질환, 가족력, 가입상품, 가입특약
    """

# ---------------------------------------------------------
# 5. 데이터 생성 및 예외 처리 로직 (API Rate Limit 대응)
# ---------------------------------------------------------
def generate_synthetic_data(total_samples, batch_size=50, max_retries=3):
    all_data = []
    batches = total_samples // batch_size
    prompt = get_prompt(batch_size)

    logger.info(f"총 {total_samples}건 데이터 생성을 시작합니다. (배치 크기: {batch_size})")

    for i in range(batches):
        retries = 0
        while retries < max_retries:
            try:
                response = model.generate_content(prompt)
                batch_data = json.loads(response.text)
                
                # 생성된 데이터가 리스트인지 검증 (데이터 무결성 체크)
                if isinstance(batch_data, list) and len(batch_data) > 0:
                    all_data.extend(batch_data)
                    logger.info(f"[{i+1}/{batches}] 배치 생성 완료 ({len(batch_data)}건)")
                    break
                else:
                    raise ValueError("응답 포맷이 올바른 JSON 배열이 아닙니다.")

            except ResourceExhausted:
                retries += 1
                wait_time = 2 ** retries * 5  # 지수 백오프 (5s, 10s, 20s)
                logger.warning(f"API 할당량 초과. {wait_time}초 대기 후 재시도... ({retries}/{max_retries})")
                time.sleep(wait_time)
            
            except json.JSONDecodeError:
                retries += 1
                logger.warning(f"JSON 파싱 에러 발생. 재시도 중... ({retries}/{max_retries})")
            
            except Exception as e:
                retries += 1
                logger.error(f"예상치 못한 오류 발생: {str(e)}")
                time.sleep(2)
        
        if retries == max_retries:
            logger.error(f"[{i+1}/{batches}] 배치 생성 최종 실패. 다음 배치로 넘어갑니다.")

    return pd.DataFrame(all_data)

# ---------------------------------------------------------
# 6. 실행 파트
# ---------------------------------------------------------
if __name__ == "__main__":
    NUM_SAMPLES = 1000
    BATCH_SIZE = 50 # 한 번의 API 호출로 생성할 데이터 수

    # 데이터 프레임 생성
    final_df = generate_synthetic_data(total_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)

    if not final_df.empty:
        # 파일 저장 (인코딩 명시 및 CSV 저장)
        output_file = "gemini_customer_data.csv"
        final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        logger.info(f"✅ 총 {len(final_df)}건의 AI 생성 데이터가 '{output_file}'로 저장되었습니다.")
    else:
        logger.error("데이터 생성에 실패하여 파일을 저장하지 못했습니다.")