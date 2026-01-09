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
      
    def extract_text_from_pdf(self, filepath):
        """PDF에서 텍스트 추출 및 NUL 문자 제거"""
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                  # \x00 (NUL) 문자를 빈 문자열로 치환하여 제거
                  text += page_text.replace('\x00', '') + "\n"
        return text
    
    def parse_kosha_guide(self, text):
        """KOSHA 가이드 구조 파싱"""
        # 빈 줄을 제외하고 순수 텍스트 줄만 리스트로 만듭니다.
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        document_id = ""
        title = ""
        id_line_index = -1
        
        # 1. document_id 추출: 'KOSHA GUIDE' 바로 다음 줄
        for i, line in enumerate(lines):
          if 'KOSHA GUIDE' in line.upper():
            if i + 1 < len(lines):
                raw_id = lines[i+1]
                # 모든 공백 제거 및 하이픈 통일
                document_id = re.sub(r'\s+', '', raw_id).replace('–', '-')
                id_line_index = i + 1
                break
            
          # 2. title 추출: document_id 다음 줄부터 '2'로 시작하는 숫자 전까지
          if id_line_index != -1:
            title_parts = []
            for j in range(id_line_index + 1, len(lines)):
                current_line = lines[j]
                
                # 중단 조건: '2'로 시작하는 숫자(예: 2020. 10. 또는 2. 적용범위 등)가 나오면 중단
                if re.match(r'^\d{4}\.', current_line) or (len(title_parts) > 0 and re.match(r'^2', current_line)):
                    break
                
                # 공단 이름이나 페이지 번호 등 불필요한 정보 스킵
                if any(x in current_line for x in ['한국산업안전보건공단', 'PAGE', 'KOSHA']):
                    continue
                    
                title_parts.append(current_line)
            
            # 3. 줄바꿈 자리에 공백 넣으며 합치기 > 남아있는 줄바꿈 제거 > 양 끝 공백 제거
            title = ' '.join(title_parts).replace('\n', ' ').strip()
            title = re.sub(r'\s+', ' ', title)
        
        return {
          "document_id": document_id,
          "title": title if title else "제목 없음",
          "metadata": {
            "guide_number": document_id,
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
          parsed_data = self.parse_kosha_guide(text)
          
          filename_title = filename[:-4].replace(parsed_data['document_id'], "").strip()
          parsed_data['title'] = filename_title
          
          search_text = self.create_search_text(parsed_data)
          
          if not parsed_data['document_id'] or 'KOSHA-' in parsed_data['document_id']:
            parsed_data['document_id'] = filename[:10].replace(" ", "")
            
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
      
    def update_document(self, document_id, full_content):
      """사용자가 수정한 데이터를 DB에 반영"""
      conn = self.get_connection()
      cur = conn.cursor()
      try:
          cur.execute("""
            UPDATE kosha_guide
            SET content = %s
            WHERE document_id = %s
          """, (
            json.dumps(full_content, ensure_ascii=False, default=self.json_default), 
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
  