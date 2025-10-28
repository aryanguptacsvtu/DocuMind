[![Typing SVG](https://readme-typing-svg.demolab.com?font=Monoton&size=80&pause=12&speed=15&color=9D00FF&center=true&vCenter=true&width=2000&height=200&lines=📚+Meet+DocuMind!;Chat+with+Your+PDFs+Effortlessly!;Ask,+Analyze,++Understand+Documents!;Powered+by+Groq+LLaMA+3.1!;RAG+Intelligence+for+Smarter+Insights!;Made+with+❤️+by+Aryan+Gupta!)](https://github.com/aryanguptacsvtu/DocuMind-RAG)

---

<h1 align="center">📚 DocuMind-RAG -- Chat with Your Documents</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/AI-Groq_LLaMA_3.1_8B-orange.svg" alt="Groq AI">
  <img src="https://img.shields.io/badge/Vector_DB-FAISS-green.svg" alt="FAISS">
  <img src="https://img.shields.io/badge/Category-Document_Chatbot-yellow.svg" alt="Category">
 

---

## 🧠 About the Project  

**DocuMind-RAG** is an intelligent **AI-powered PDF chatbot** that lets you upload multiple documents and **chat directly with their content**.  
It uses **Retrieval-Augmented Generation (RAG)** with **LangChain**, **FAISS**, and **Groq LLaMA 3.1** to deliver accurate, context-aware answers from your files.

Whether you're analyzing research papers, study material, or business reports — **DocuMind** transforms your PDFs into interactive conversations.


---

## Core Features

✅ Upload and process multiple PDF files  
✅ Extract, chunk, and embed text using `HuggingFace` transformers  
✅ Store vector embeddings efficiently with `FAISS`  
✅ Query your PDFs conversationally using `LLaMA 3-8B (Groq API)`  
✅ Memory-aware responses with contextual follow-ups  
✅ Clean Streamlit UI for real-time chatting  

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Backend & RAG:** [LangChain](https://www.langchain.com/)
* **LLM:** [Groq](https://groq.com/) (Llama 3.1 8B Instant)
* **Embeddings:** [Hugging Face](https://huggingface.co/sentence-transformers) (`all-MiniLM-L6-v2`)
* **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss) (from Meta)
* **PDF Parsing:** [PyPDF2](https://pypdf2.readthedocs.io/en/latest/)

---
## 📦 Setup & Installation

Follow these steps to run DocuMind on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/DocuMind.git
cd DocuMind
```

### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

```Bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
You can install all Python packages using the provided requirements.txt file.

```Bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
The application uses an API key for the Groq LLM.

1. Create a file named .env in the root of your project directory.

2. Add your Groq API key to this file:
`GROQ_API_KEY="your-api-key-here"`

### 5. Run the Streamlit App
Once everything is installed, launch the app from your terminal:

```Bash
streamlit run frontend.py
```

Your browser should automatically open to the application.

---

## ▶️ Usage

1. Launch the application using the command above.

2. Use the sidebar to upload one or more PDF files you wish to chat with.

3. Click the "Process" button and wait for the "Documents processed!" success message.

4. The chat input box at the bottom of the page will become active.

5. Start asking your questions!

---

## 👨‍💻 Author

**Aryan Gupta**  
📍 Bhilai, Chhattisgarh  
🔗 [GitHub Profile](https://github.com/aryanguptacsvtu)

---
## ⭐ Support

If you like this project, leave a ⭐ and share it with others!
