import psycopg2
from psycopg2.extras import Json, RealDictCursor
from psycopg2.extensions import register_adapter, AsIs
import PyPDF2
import re
import os
from datetime import datetime
import json
import httpx
from typing import List
import pdfplumber
from datetime import datetime, date
# OpenAI 사용 시 주석 해제
# from openai import OpenAI

# 임베딩 설정
EMBEDDING_BASE_URL = os.getenv('OLLAMA_BASE_URL')
EMBEDDING_MODEL = "bona/bge-m3-korean:latest"

class PDFProcessor:
    def __init__(self, db_config, openai_api_key=None):
        self.db_config = db_config
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # OpenAI 클라이언트 초기화 (API 키가 있는 경우)
        # if openai_api_key:
        #     self.openai_client = OpenAI(api_key=openai_api_key)
        # else:
        #     self.openai_client = None
        
        # pgvector를 위한 어댑터 등록
        register_adapter(list, self._adapt_list_to_vector)
    
    def _adapt_list_to_vector(self, lst):
        """리스트를 PostgreSQL vector 타입으로 변환"""
        return AsIs(f"'[{','.join(map(str, lst))}]'")
    
    def get_connection(self):
        return psycopg2.connect(**self.db_config)
    
    def create_embedding(self, text: str) -> List[float]:
      """단일 텍스트를 Ollama를 통해 임베딩하여 벡터(List[float])로 반환합니다."""
      payload = {
        "model": EMBEDDING_MODEL,
        "input": [text], # API는 리스트 형태를 받음
      }
      try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{EMBEDDING_BASE_URL}/api/embed", json=payload)
            resp.raise_for_status()
            data = resp.json()
            # 입력이 하나이므로 첫 번째 결과 반환
            return data["embeddings"][0]
      except Exception as e:
        print(f"❌ 임베딩 API 호출 실패: {e}")
        return []
    
    def create_dummy_embedding(self, dimension=1024):
        """테스트용 더미 임베딩 생성"""
        import random
        return [random.random() for _ in range(dimension)]
    
    def _parse_sections(self, text):
      """불필요한 헤더/푸터를 제거하고 띄어쓰기와 줄바꿈을 보존하는 파싱 로직"""
      sections = []
      current_section = None
      lines = text.split('\n')
      # 제거할 노이즈 패턴들 정의
      noise_patterns = [
        r'-\s*\d+\s*-',                          # 페이지 번호 (- 3 -)
        r'[A-Z]\s*-\s*\d+\s*-\s*\d{4}',          # 문서 번호 (G - 72 - 2011)
        r'KOSHA\s+GUIDE',                        # KOSHA GUIDE 문구
        r'한국산업안전보건공단'                  # 기관명
      ]
      # 패턴들을 하나로 합침
      combined_noise_re = re.compile('|'.join(noise_patterns))
      
      for line in lines:
          line_strip = line.strip()
          if not line_strip: continue
          
          # [추가] 노이즈 패턴과 완벽히 일치하거나, 해당 줄이 노이즈라면 건너뜀
          # 특히 페이지 번호나 문서 번호만 있는 줄을 제거합니다.
          if combined_noise_re.fullmatch(line_strip) or line_strip in ["KOSHA GUIDE", "한국산업안전보건공단"]:
            continue
          
          # 섹션 번호 패턴 (예: 1. 목 적) [cite: 33]
          main_match = re.match(r'^(\d+)\.\s*(.+)$', line_strip)
          if main_match and len(main_match.group(1)) <= 2:
              if current_section:
                  sections.append(current_section)
              current_section = {
                  "number": main_match.group(1),
                  "title": main_match.group(2).strip(),
                  "content": "",
                  "subsections": []
              }
              continue
          
          if current_section:
              # ' ' 대신 '\n'을 추가하여 문장 간 띄어쓰기 보존
              current_section["content"] += line_strip + "\n"
      
      if current_section:
          sections.append(current_section)
      return sections
    
    # def extract_table_as_markdown(self, page, table_settings=None):
    #   """페이지 내 테이블을 추출하여 마크다운 문자열로 변환하되, 헤더성 테이블은 제외"""
    #   tables = page.extract_tables(table_settings=table_settings)
    #   if not tables: return ""
    #   md_output = ""
    #   exclude_keywords = ["KOSHA GUIDE", "한국산업안전보건공단"] # 제외하고 싶은 헤더 키워드
    #   for table in tables:
    #     # 빈 표나 헤더만 있는 표 방지
    #     all_cells = [str(cell) for row in table for cell in row if cell]
    #     all_text = "".join(all_cells)
    #     # 테이블의 모든 텍스트를 하나로 합쳐서 키워드 체크
    #     if not all_text or any(kw in all_text for kw in exclude_keywords):
    #       continue
        
    #     # 마크다운 표 생성 로직
    #     formatted_rows = []
    #     for row in table:
    #       # 셀 내 줄바꿈 제거 및 None 처리
    #       clean_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
    #       formatted_rows.append(clean_row)
    #     if len(formatted_rows) < 1: continue
        
    #     md_table = "\n"
    #     for i, row in enumerate(formatted_rows):
    #       md_table += "| " + " | ".join(row) + " |\n"
    #       if i == 0: # 마크다운 표 구분선
    #         md_table += "| " + " | ".join(["---"] * len(row)) + " |\n"
    #     md_output += md_table + "\n"
          
    #   return md_output
  
    def extract_table_as_markdown(self, page, settings):
      table_data = page.extract_table(settings)
      if not table_data:
        return ""
      md_out = ""
      for i, row in enumerate(table_data):
        # None 값을 공백으로 치환하고 각 셀의 줄바꿈 제거
        clean_row = [str(cell or "").replace("\n", " ").strip() for cell in row]
        md_out += "| " + " | ".join(clean_row) + " |\n"
        # 헤더 밑에 구분선 추가
        if i == 0:
            md_out += "| " + " | ".join(["---"] * len(clean_row)) + " |\n"
      return md_out
  
    def extract_text_from_pdf(self, filepath):
      """PDF에서 텍스트와 테이블을 분리 추출하여 중복 없이 결합"""
      text = ""
      # 표 인식을 위한 정밀 설정 (부록 1과 같은 복잡한 표 대응)
      table_settings = {
        "vertical_strategy": "lines",   # 선을 기준으로 열 구분
        "horizontal_strategy": "lines", # 선을 기준으로 행 구분
        # "intersection_tolerance": 15,     # 교차점 인식 범위 확대
        "snap_tolerance": 3,            # 떨어진 선을 붙여서 인식하는 허용치
        "join_tolerance": 3,
        "edge_min_length": 15,          # 너무 짧은 선은 무시
      }
      with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
          # 0. 페이지 상하단 노이즈 제거 (머리말/꼬리말 제외 영역 설정)
          # 페이지 전체 높이의 상단 10%, 하단 10%를 제외한 영역만 사용
          page_height = page.height
          content_area = (0, page_height * 0.1, page.width, page_height * 0.9)
          cropped_page = page.within_bbox(content_area)
          # 1. 현재 페이지에서 테이블들을 찾고 좌표(bbox) 리스트를 생성합니다.
          # 이 부분이 정의되어야 아래 filter에서 에러가 나지 않습니다.
          tables = cropped_page.find_tables(table_settings=table_settings)
          table_bboxes = [t.bbox for t in tables]
          # 2. 필터 함수 정의 (table_bboxes를 참조하여 테이블 안에 있는지 판단)
          def is_not_in_table(obj):
            # 객체(글자 등)의 좌표 추출
            obj_bbox = pdfplumber.utils.obj_to_bbox(obj)
            for t_bbox in table_bboxes:
              # 글자의 좌표가 테이블 영역(t_bbox) 안에 포함되는지 검사 : 미세한 오차를 줄이기 위해 +-1 정도의 여유를 둡니다.
              if (obj_bbox[0] >= t_bbox[0] - 1 and 
                obj_bbox[1] >= t_bbox[1] - 1 and 
                obj_bbox[2] <= t_bbox[2] + 1 and 
                obj_bbox[3] <= t_bbox[3] + 1):
                  return False  # 테이블 안에 있으므로 제거 대상
            return True  # 테이블 밖에 있으므로 유지 대상

          # 3. 테이블 영역이 제거된 깨끗한 텍스트 추출
          clean_text = cropped_page.filter(is_not_in_table).extract_text()
          # 4. 테이블을 마크다운으로 별도 추출
          md_tables = self.extract_table_as_markdown(cropped_page, table_settings)
          # 5. 결합 (순서는 텍스트 후 테이블)
          if clean_text:
            text += clean_text.replace('\x00', '') + "\n"
          if md_tables:
            text += md_tables + "\n"
      return text
    
    def parse_kosha_guide(self, text, filename):
      """KOSHA 가이드 구조 파싱"""
      # 1. 파일명에서 확장자 제거
      base_name = os.path.splitext(filename)[0]
      # 2. 정규표현식으로 ID와 타이틀 분리 # 예: A-180-2020, G-1-2023 등
      match = re.match(r'^([A-Z]-\d+-\d+)\s*(.*)$', base_name)
      filename_id = ""
      filename_title = ""
      if match:
        filename_id = match.group(1).strip()
        filename_title = match.group(2).strip()
      else: # 파일명 패턴이 안 맞을 경우 전체를 타이틀로 일단 간주
        filename_id = base_name
        filename_title = base_name
          
      return {
        "document_id": filename_id,
        "title": filename_title,
        "metadata": {
          "guide_number": filename_id,
          "publication_date": self._extract_date(text),
          "publisher": "한국산업안전보건공단",
          "authors": self._extract_authors(text),
          "revision_history": self._extract_revision_history(text),
          "related_regulation": self._extract_related_regulation(text)
        },
        "sections": self._parse_sections(text),
        "forms": self._parse_forms(text)
      }
    
    def _extract_date(self, text):
        patterns = [
            r'(\d{4})\s*\.\s*(\d{1,2})\s*\.',
            r'(\d{4})\s*년\s*(\d{1,2})\s*월',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return f"{match.group(1)}-{match.group(2).zfill(2)}"
        return None
    
    def _extract_authors(self, text):
      author_lines = []
      in_section = False
      for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        
        if '작성자' in line:
          # 같은 줄에 이름이 있는 경우 (예: ○ 작성자 : 대한기계학회 서상호)
          content_after_colon = line.split(':', 1)[-1].strip()
          if content_after_colon:
            return content_after_colon
          in_section = True
          continue
            
        if in_section:
          if line.startswith('○') or '제·개정' in line:
            break
          author_lines.append(line)
                
      return "\n".join(author_lines).strip()
    
    def _extract_revision_history(self, text):
      history_lines = []
      in_section = False
      for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        if '개정 경과' in line:
          in_section = True
          continue
        if in_section:
          if line.startswith('○') or '관련규격' in line:
            break
          history_lines.append(line)
      return "\n".join(history_lines).strip()
    
    def _extract_related_regulation(self, text):
      reg_lines = []
      in_section = False
      for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        if '관련법규' in line:
          in_section = True
          continue
        if in_section:
          if line.startswith('○') or '기술지침' in line:
            break
          reg_lines.append(line)
      return "\n".join(reg_lines).strip()
    
    def _parse_forms(self, text):
        forms = []
        pattern = r'<별지\s*서식\s*(\d+)>'
        matches = list(re.finditer(pattern, text))
        
        for i, match in enumerate(matches):
            start_pos = match.end()
            end_pos = matches[i+1].start() if i+1 < len(matches) else len(text)
            content = text[start_pos:end_pos].strip()[:1000]
            
            if len(content) > 10:
                forms.append({
                    "form_number": match.group(1),
                    "title": match.group(0),
                    "content": content
                })
        
        return forms
    
    def create_search_text(self, content_dict):
        """검색용 텍스트 생성"""
        parts = []
        if content_dict.get('title'):
            parts.append(content_dict['title'])
        if content_dict.get('document_id'):
            parts.append(content_dict['document_id'])
        for section in content_dict.get('sections', []):
            if section.get('title'):
                parts.append(section['title'])
            if section.get('content'):
                parts.append(section['content'][:500])
        return ' '.join(parts)
    
    def process_and_save(self, filepath, filename):
        """PDF 처리 및 DB 저장 Args: filepath: PDF 파일 경로"""
        try:
          # filename = os.path.basename(filepath)
          text = self.extract_text_from_pdf(filepath)
          parsed_data = self.parse_kosha_guide(text, filename)
          search_text = self.create_search_text(parsed_data)
          
          # 실제 OpenAI 임베딩 생성 (위에서 주석 해제 필요)
          embedding_vector = self.create_embedding(search_text)
          
          # 테스트용 더미 임베딩
          if embedding_vector is None:
              print("[INFO] 더미 임베딩 생성 (테스트용)")
              embedding_vector = self.create_dummy_embedding()
          
          conn = self.get_connection()
          cur = conn.cursor()
          
          cur.execute(
              "SELECT id FROM kosha_guide WHERE document_id = %s",
              (parsed_data['document_id'],)
          )
          existing = cur.fetchone()
          
          if existing:
              # 업데이트
              if embedding_vector:
                  cur.execute("""
                      UPDATE kosha_guide 
                      SET title = %s,
                          content = %s,
                          search_embedding = to_tsvector('korean', %s),
                          embedding = %s::vector,
                          updated_at = CURRENT_TIMESTAMP
                      WHERE document_id = %s
                      RETURNING id
                  """, (
                      parsed_data['title'],
                      Json(parsed_data),
                      search_text,
                      embedding_vector,
                      parsed_data['document_id']
                  ))
              else:
                  cur.execute("""
                      UPDATE kosha_guide 
                      SET title = %s,
                          content = %s,
                          search_embedding = to_tsvector('korean', %s),
                          updated_at = CURRENT_TIMESTAMP
                      WHERE document_id = %s
                      RETURNING id
                  """, (
                      parsed_data['title'],
                      Json(parsed_data),
                      search_text,
                      parsed_data['document_id']
                  ))
          else:
              # 신규 삽입
              if embedding_vector:
                  cur.execute("""
                      INSERT INTO kosha_guide 
                      (document_id, title, content, search_embedding, embedding)
                      VALUES (%s, %s, %s, to_tsvector('korean', %s), %s::vector)
                      RETURNING id
                  """, (
                      parsed_data['document_id'],
                      parsed_data['title'],
                      Json(parsed_data),
                      search_text,
                      embedding_vector
                  ))
              else:
                  cur.execute("""
                      INSERT INTO kosha_guide 
                      (document_id, title, content, search_embedding)
                      VALUES (%s, %s, %s, to_tsvector('korean', %s))
                      RETURNING id
                  """, (
                      parsed_data['document_id'],
                      parsed_data['title'],
                      Json(parsed_data),
                      search_text
                  ))
          
          result_id = cur.fetchone()[0]
          conn.commit()
          
          print(f"[SUCCESS] DB 저장 완료 - ID: {result_id}")
          print(f"[INFO] Embedding 사용: {embedding_vector}")
          
          return {
              'id': result_id,
              'document_id': parsed_data['document_id'],
              'title': parsed_data['title'],
              'sections_count': len(parsed_data['sections']),
              'has_embedding': embedding_vector is not None
          }
            
        except Exception as e:
            print(f"[ERROR] {e}")
            if 'conn' in locals():
                conn.rollback()
            raise e
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()
    
    def get_all_documents(self):
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute("""
                SELECT id, document_id, title, created_at, updated_at
                FROM kosha_guide
                ORDER BY created_at DESC
            """)
            return cur.fetchall()
        finally:
            cur.close()
            conn.close()
    
    def get_document_by_id(self, document_id):
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute("""
                SELECT * FROM kosha_guide
                WHERE document_id = %s
            """, (document_id,))
            return cur.fetchone()
        finally:
            cur.close()
            conn.close()
    
    def search_documents(self, query):
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute("""
                SELECT id, document_id, title,
                       ts_rank(search_embedding, to_tsquery('korean', %s)) as rank
                       , created_at, updated_at
                FROM kosha_guide
                WHERE search_embedding @@ to_tsquery('korean', %s)
                ORDER BY rank DESC
                LIMIT 20
            """, (query, query))
            
            results = cur.fetchall()
            if not results:
                cur.execute("""
                    SELECT id, document_id, title, created_at, updated_at
                    FROM kosha_guide
                    WHERE title ILIKE %s OR content::text ILIKE %s
                    LIMIT 20
                """, (f'%{query}%', f'%{query}%'))
                results = cur.fetchall()
            
            return results
        finally:
            cur.close()
            conn.close()
    
    def similarity_search(self, query_text, limit=10):
        """임베딩 기반 유사도 검색"""
        # 쿼리 텍스트를 임베딩으로 변환
        query_embedding = self.create_embedding(query_text)
        
        if query_embedding is None:
            print("[WARNING] 임베딩 미사용, 일반 검색으로 대체")
            return self.search_documents(query_text)
        
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # 코사인 유사도로 검색
            cur.execute("""
                SELECT 
                    id, 
                    document_id, 
                    title,
                    1 - (embedding <=> %s::vector) as similarity
                FROM kosha_guide
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, limit))
            
            return cur.fetchall()
        finally:
            cur.close()
            conn.close()
      
    def update_document(self, document_id, full_content, title):
      """사용자가 수정한 데이터를 DB에 반영"""
      conn = self.get_connection()
      cur = conn.cursor()
      try:
          cur.execute("""
            UPDATE kosha_guide
            SET content = %s, title = %s
            WHERE document_id = %s
          """, (
            json.dumps(full_content, ensure_ascii=False, default=self.json_default), 
            title,
            document_id
          ))
          conn.commit()
          print(f"[SUCCESS] 사용자가 수정한 데이터를 DB에 반영 완료 : {cur.rowcount > 0}")
          return cur.rowcount > 0
      except Exception as e:
        print(f"Update error: {e}")
        conn.rollback() # 에러 발생 시 되돌립니다.
        return False
      finally:
        cur.close()
        conn.close()
        
    def json_default(self, value):
      if isinstance(value, (datetime, date)):
          return value.isoformat() # '2024-05-20T10:00:00' 형태로 변환
      raise TypeError(f'Object of type {value.__class__.__name__} is not JSON serializable')
  