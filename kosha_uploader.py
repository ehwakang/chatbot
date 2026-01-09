#!/usr/bin/env python3
"""
KOSHA 가이드 PDF 업로드 시스템 - 커맨드라인 버전
GUI 없이 터미널에서 실행 가능
"""

import argparse
import pdfplumber
import psycopg2
import json
import re
import os
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path


class KoshaGuideParser:
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF에서 텍스트 추출"""
        print(f"📄 PDF 텍스트 추출 중: {pdf_path}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    print(f"   페이지 {i}/{len(pdf.pages)} 처리 완료", end='\r')
                print()  # 줄바꿈
                return text
        except Exception as e:
            raise Exception(f"PDF 파싱 실패: {str(e)}")
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """메타데이터 추출"""
        print("🔍 메타데이터 추출 중...")
        metadata = {}
        
        # 가이드 번호
        guide_patterns = [
            r'KOSHA\s+GUIDE\s*\n\s*([P|C|M|E|H|G|W]\s*-?\s*\d+\s*[–-]\s*\d{4})',
            r'([P|C|M|E|H|G|W]\s*-\s*\d+\s*–\s*\d{4})',
        ]
        
        for pattern in guide_patterns:
            guide_match = re.search(pattern, text, re.IGNORECASE)
            if guide_match:
                metadata["guide_number"] = guide_match.group(1).strip().replace('–', '-').replace(' ', '')
                print(f"   ✓ 가이드 번호: {metadata['guide_number']}")
                break
        
        # 제목 추출
        title_patterns = [
            r'[–-]\s*\d{4}\s*\n\s*(.+?)\s*\n',
            r'GUIDE\s*\n\s*[P|C|M|E|H|G|W]\s*-\s*\d+\s*[–-]\s*\d{4}\s*\n\s*(.+?)\s*\n',
        ]
        
        for pattern in title_patterns:
            title_match = re.search(pattern, text)
            if title_match:
                title = title_match.group(1).strip()
                title = ' '.join(title.split())
                metadata["title"] = title
                print(f"   ✓ 제목: {title}")
                break
        
        # 발행일
        date_match = re.search(r'(\d{4})\s*\.\s*(\d{1,2})\s*\.', text)
        if date_match:
            year = date_match.group(1)
            month = date_match.group(2).zfill(2)
            metadata["publication_date"] = f"{year}-{month}"
            print(f"   ✓ 발행일: {metadata['publication_date']}")
        
        # 발행기관
        metadata["publisher"] = "한국산업안전보건공단"
        
        # 작성자 및 개정자
        authors = []
        author_patterns = [r'작성자:\s*(.+)', r'개정자:\s*(.+)']
        for pattern in author_patterns:
            matches = re.findall(pattern, text)
            authors.extend([m.strip() for m in matches if m.strip()])
        
        if authors:
            metadata["authors"] = authors
            print(f"   ✓ 작성자/개정자: {len(authors)}명")
        
        # 제개정 경과
        revision_history = []
        revision_match = re.search(r'제·개정 경과\s*\n(.*?)(?=¡|관련 규격|$)', text, re.DOTALL)
        if revision_match:
            revision_text = revision_match.group(1)
            revisions = re.findall(r'-\s*(.+)', revision_text)
            revision_history = [rev.strip() for rev in revisions if rev.strip() and len(rev.strip()) > 5]
        
        if revision_history:
            metadata["revision_history"] = revision_history
            print(f"   ✓ 개정 이력: {len(revision_history)}건")
        
        # 관련 규격
        related_standards = []
        standards_match = re.search(r'관련 규격 및 자료\s*\n(.*?)(?=¡|기술지침의|$)', text, re.DOTALL)
        if standards_match:
            standards_text = standards_match.group(1)
            standards = re.findall(r'-\s*(.+)', standards_text)
            related_standards = [std.strip() for std in standards if std.strip() and len(std.strip()) > 5]
        
        if related_standards:
            metadata["related_standards"] = related_standards
            print(f"   ✓ 관련 규격: {len(related_standards)}개")
        
        return metadata
    
    def extract_subsections(self, content: str) -> List[Dict[str, Any]]:
        """하위 섹션 추출"""
        subsections = []
        
        # 패턴 1: 숫자.숫자 형식
        decimal_pattern = r'(\d+\.\d+)\s+([^\n]+)'
        decimal_matches = re.finditer(decimal_pattern, content)
        
        for match in decimal_matches:
            number = match.group(1)
            title = match.group(2).strip()
            
            if len(title) > 3:
                subsections.append({
                    "number": number,
                    "title": title[:150]
                })
        
        # 패턴 2: 괄호 번호 형식
        if not subsections:
            paren_patterns = [
                r'\((\d+)\)\s*([^\n]+)',
                r'\(([가-힣])\)\s*([^\n]+)',
            ]
            
            for pattern in paren_patterns:
                paren_matches = re.finditer(pattern, content)
                for match in paren_matches:
                    number = match.group(1)
                    text = match.group(2).strip()
                    
                    if len(text) > 3:
                        subsections.append({
                            "number": f"({number})",
                            "content": text[:200]
                        })
                
                if subsections:
                    break
        
        return subsections
    
    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """본문 섹션 추출"""
        print("📑 섹션 구조 분석 중...")
        sections = []
        
        section_pattern = r'\n(\d+)\.\s+([^\n]+)\n(.*?)(?=\n\d+\.\s+[^\n]+\n|<별지|$)'
        matches = re.finditer(section_pattern, text, re.DOTALL)
        
        for match in matches:
            section_num = match.group(1)
            section_title = match.group(2).strip()
            section_content = match.group(3).strip()
            
            if len(section_content) < 10:
                continue
            
            subsections = self.extract_subsections(section_content)
            
            section_data = {
                "number": section_num,
                "title": section_title,
                "content": section_content if len(section_content) < 500 else section_content[:500] + "...",
                "subsections": subsections
            }
            
            sections.append(section_data)
            print(f"   ✓ 섹션 {section_num}: {section_title} (하위섹션: {len(subsections)}개)")
        
        return sections
    
    def extract_forms(self, text: str) -> List[Dict[str, Any]]:
        """별지 서식 추출"""
        forms = []
        
        form_pattern = r'<별지\s*서식\s*(\d+)>\s*\n\s*([^\n]+)'
        matches = re.finditer(form_pattern, text)
        
        for match in matches:
            form_num = match.group(1)
            form_title = match.group(2).strip()
            
            forms.append({
                "form_number": form_num,
                "title": form_title
            })
        
        if forms:
            print(f"📋 별지 서식: {len(forms)}개 발견")
            for form in forms:
                print(f"   ✓ 별지 서식 {form['form_number']}: {form['title']}")
        
        return forms
    
    def parse_kosha_guide(self, text: str) -> Dict[str, Any]:
        """KOSHA 가이드 텍스트를 JSON으로 파싱"""
        
        metadata = self.extract_metadata(text)
        sections = self.extract_sections(text)
        forms = self.extract_forms(text)
        
        return {
            "document_id": metadata.get("guide_number", "UNKNOWN"),
            "title": metadata.get("title", "제목 없음"),
            "metadata": metadata,
            "sections": sections,
            "forms": forms
        }
    
    def save_to_database(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터베이스에 저장"""
        print("\n💾 데이터베이스 저장 중...")
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # 중복 체크
            cur.execute(
                "SELECT id, title FROM kosha_guide WHERE document_id = %s",
                (parsed_data["document_id"],)
            )
            existing = cur.fetchone()
            
            if existing:
                # 업데이트
                print(f"   ℹ️  기존 문서 발견 (ID: {existing[0]})")
                cur.execute("""
                    UPDATE kosha_guide 
                    SET title = %s,
                        content = %s,
                        updated_at = NOW()
                    WHERE document_id = %s
                    RETURNING id, document_id, title
                """, (
                    parsed_data["title"],
                    json.dumps(parsed_data, ensure_ascii=False),
                    parsed_data["document_id"]
                ))
                result = cur.fetchone()
                message = f"기존 문서(ID: {existing[0]})가 업데이트되었습니다."
            else:
                # 새로 생성
                print(f"   ℹ️  새 문서 생성")
                cur.execute("""
                    INSERT INTO kosha_guide (document_id, title, content)
                    VALUES (%s, %s, %s)
                    RETURNING id, document_id, title
                """, (
                    parsed_data["document_id"],
                    parsed_data["title"],
                    json.dumps(parsed_data, ensure_ascii=False)
                ))
                result = cur.fetchone()
                message = "새 문서가 성공적으로 저장되었습니다."
            
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"   ✅ {message}")
            
            return {
                "id": result[0],
                "document_id": result[1],
                "title": result[2],
                "sections_count": len(parsed_data.get("sections", [])),
                "forms_count": len(parsed_data.get("forms", [])),
                "message": message
            }
        
        except psycopg2.Error as e:
            raise Exception(f"데이터베이스 오류: {str(e)}")
        except Exception as e:
            raise Exception(f"저장 실패: {str(e)}")
    
    def process_file(self, pdf_path: str, output_json: str = None) -> Dict[str, Any]:
        """PDF 파일 처리"""
        print("=" * 80)
        print(f"KOSHA 가이드 PDF 파싱 시작")
        print("=" * 80)
        print()
        
        # 1. 텍스트 추출
        text = self.extract_text_from_pdf(pdf_path)
        print(f"   총 {len(text):,} 문자 추출\n")
        
        # 2. 파싱
        parsed_data = self.parse_kosha_guide(text)
        
        # 3. JSON 파일로 저장 (옵션)
        if output_json:
            print(f"\n💾 JSON 파일 저장 중: {output_json}")
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=2)
            print(f"   ✅ JSON 파일 저장 완료")
        
        # 4. 데이터베이스 저장
        result = self.save_to_database(parsed_data)
        
        # 5. 결과 출력
        print("\n" + "=" * 80)
        print("처리 완료!")
        print("=" * 80)
        print(f"\n📊 결과 요약:")
        print(f"   - 데이터베이스 ID: {result['id']}")
        print(f"   - 문서 ID: {result['document_id']}")
        print(f"   - 제목: {result['title']}")
        print(f"   - 섹션 수: {result['sections_count']}개")
        print(f"   - 별지 서식: {result['forms_count']}개")
        print(f"   - 상태: {result['message']}")
        print()
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description='KOSHA 가이드 PDF를 파싱하여 데이터베이스에 저장합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 단일 파일 처리
  python cli_kosha_uploader.py input.pdf
  
  # JSON 파일도 함께 저장
  python cli_kosha_uploader.py input.pdf --output output.json
  
  # 환경변수 파일 지정
  python cli_kosha_uploader.py input.pdf --env-file .env.production
  
  # DB 설정 직접 지정
  python cli_kosha_uploader.py input.pdf --db-host localhost --db-name kosha_db
        """
    )
    
    parser.add_argument('pdf_file', help='처리할 PDF 파일 경로')
    parser.add_argument('-o', '--output', help='출력 JSON 파일 경로 (선택사항)')
    parser.add_argument('--env-file', default='.env', help='환경변수 파일 경로 (기본값: .env)')
    
    # DB 설정 (명령줄 인자로도 지정 가능)
    parser.add_argument('--db-host', help='데이터베이스 호스트')
    parser.add_argument('--db-port', type=int, help='데이터베이스 포트')
    parser.add_argument('--db-name', help='데이터베이스 이름')
    parser.add_argument('--db-user', help='데이터베이스 사용자')
    parser.add_argument('--db-password', help='데이터베이스 비밀번호')
    
    args = parser.parse_args()
    
    # PDF 파일 존재 확인
    if not os.path.exists(args.pdf_file):
        print(f"❌ 오류: 파일을 찾을 수 없습니다: {args.pdf_file}")
        return 1
    
    # DB 설정 로드
    db_config = {}
    
    # 1. .env 파일에서 로드 시도
    if os.path.exists(args.env_file):
        try:
            from dotenv import load_dotenv
            load_dotenv(args.env_file)
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD')
            }
        except ImportError:
            print("⚠️  경고: python-dotenv가 설치되지 않았습니다. 명령줄 인자를 사용하세요.")
    
    # 2. 명령줄 인자로 덮어쓰기
    if args.db_host:
        db_config['host'] = args.db_host
    if args.db_port:
        db_config['port'] = args.db_port
    if args.db_name:
        db_config['database'] = args.db_name
    if args.db_user:
        db_config['user'] = args.db_user
    if args.db_password:
        db_config['password'] = args.db_password
    
    # DB 설정 검증
    required_keys = ['host', 'database', 'user', 'password']
    missing_keys = [key for key in required_keys if not db_config.get(key)]
    
    if missing_keys:
        print(f"❌ 오류: 다음 데이터베이스 설정이 누락되었습니다: {', '.join(missing_keys)}")
        print(f"   .env 파일을 생성하거나 명령줄 인자로 지정하세요.")
        return 1
    
    # 처리 시작
    try:
        uploader = KoshaGuideParser(db_config)
        uploader.process_file(args.pdf_file, args.output)
        return 0
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())