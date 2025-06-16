from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from backend import retrieve_relevant_chunks, generate_answer, vectorstore


# Your RAG logic functions
def retrieve_context(query, vectorstore=vectorstore):
    # Use your vector database to retrieve relevant chunks
    chunks = retrieve_relevant_chunks(query, vectorstore=vectorstore)
    return "السياق المرتبط بالسؤال: " + query + "؟ \n" + str(chunks), chunks

def generate_answer_interface(context, query):
    # Use your LLM to generate answer
    answer = generate_answer(query , context)
    return "الجواب الذي تم إنشاؤه للسؤال: " + query + '؟ \n' + answer

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/rag", methods=["POST"])
def rag():
    data = request.get_json()
    query = data.get("query", "")
    meta, context = retrieve_context(query)
    answer = generate_answer_interface(context, query)
    return jsonify({"context": meta, "answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
