from flask import Flask, render_template, request, jsonify
from langchain_llama2 import llama2_main_function

app = Flask(__name__)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'txt'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})


    file = request.files['file']
    content = file.read()
    result = llama2_main_function(content)
    print("printing result.............")
    print(result)
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and file.filename.endswith('.txt'):
        content = file.read()
        return jsonify({"result": content.decode('utf-8')})
    else:
        return jsonify({"error": "Invalid file format. Please upload a .txt file"})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        content = file.read()
        return render_template('index.html', content=content)
    else:
        return "Invalid file format. Please upload a .txt file."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
