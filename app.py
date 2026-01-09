from flask import Flask, render_template, request, jsonify, make_response, json
import os
from werkzeug.utils import secure_filename
from processor import PDFProcessor
from datetime import datetime, date

app = Flask(__name__)
# 한글 깨짐 방지: JSON 응답 시 유니코드로 변환하지 않고 그대로 출력
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# uploads 폴더가 없으면 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# PostgreSQL 연결 정보
DB_CONFIG = {
  'host': os.getenv("DB_HOST"),
  'database': os.getenv("DB_NAME"),
  'user': os.getenv("DB_USER"),
  'password': os.getenv("DB_PASSWORD"),
  'port': os.getenv("DB_PORT"),
  'client_encoding': 'UTF8'
}

pdf_processor = PDFProcessor(DB_CONFIG)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'PDF 파일만 업로드 가능합니다.'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # PDF 처리 및 DB 저장
        result = pdf_processor.process_and_save(filepath, file.filename)
        
        # 업로드된 파일 삭제 (선택사항)
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': 'PDF가 성공적으로 처리되었습니다.',
            'data': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'처리 중 오류가 발생했습니다: {str(e)}'
        }), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    try:
        documents = pdf_processor.get_all_documents()
        return jsonify({
            'success': True,
            'documents': documents
        }), 200
    except Exception as e:
        return jsonify({
            'error': f'문서 목록 조회 중 오류가 발생했습니다: {str(e)}'
        }), 500

def json_default(value):
  if isinstance(value, (datetime, date)):
      return value.isoformat() # '2024-05-20T10:00:00' 형태로 변환
  raise TypeError(f'Object of type {value.__class__.__name__} is not JSON serializable')
  
@app.route('/documents/<document_id>', methods=['GET'])
def get_document(document_id):
    try:
        document = pdf_processor.get_document_by_id(document_id)
        if document:
          response_content = json.dumps({
              'success': True,
              'document': document
          }, ensure_ascii=False, default=json_default, indent=4)
          res = make_response(response_content)
          res.headers['Content-Type'] = 'application/json; charset=utf-8'
          return res, 200
        else:
            return jsonify({
                'error': '문서를 찾을 수 없습니다.'
            }), 404
    except Exception as e:
        return jsonify({
            'error': f'문서 조회 중 오류가 발생했습니다: {str(e)}'
        }), 500
        
@app.route('/documents/<document_id>', methods=['PUT'])
def update_document(document_id):
    try:
      data = request.json

      # 1. 기존 DB 데이터 가져오기
      doc = pdf_processor.get_document_by_id(document_id)
      if not doc:
          return jsonify({'success': False, 'error': '문서를 찾을 수 없습니다.'}), 404

      full_content = doc['content']

      # 2. 메타데이터 업데이트 (작성자, 경과, 법규 등)
      if 'metadata' in data:
        # 기존 metadata 딕셔너리에 새로운 내용을 덮어씁니다.
        full_content['metadata'].update(data['metadata'])
            
      # 3. 섹션 내용 업데이트
      new_sections = data.get('sections', [])
      for update_item in new_sections:
        idx = update_item['index']
        if 0 <= idx < len(full_content['sections']):
          full_content['sections'][idx]['content'] = update_item['content']

      # 4. DB에 저장 
      success = pdf_processor.update_document(document_id, full_content)

      if success:
          return jsonify({'success': True})
      else:
          return jsonify({'success': False, 'error': 'DB 저장 실패'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
      
@app.route('/search', methods=['GET'])
def search_documents():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': '검색어를 입력해주세요.'}), 400
    
    try:
        results = pdf_processor.search_documents(query)
        return jsonify({
            'success': True,
            'results': results
        }), 200
    except Exception as e:
        return jsonify({
            'error': f'검색 중 오류가 발생했습니다: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)