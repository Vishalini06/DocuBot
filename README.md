# DocuBot
Talk to your PDFs like never before — powered by HuggingFace, LangChain, and Streamlit.


An AI-powered PDF chatbot that allows users to upload one or more PDFs and ask questions based on the document content. Built with Streamlit, LangChain, HuggingFace Transformers, and Chroma vector database.

---

## 🔍 Features

- Upload and process multiple PDFs at once
- Ask natural language questions about the documents
- Retrieves context-aware answers from the PDF content
- Uses HuggingFace Transformers for language modeling
- Vector storage with Chroma DB and sentence-transformer embeddings
- Clean, interactive UI with Streamlit

---

## 🚀 Demo

![PDF Chatbot Demo](https://user-images.githubusercontent.com/your-demo-image.gif)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: LangChain, HuggingFace Transformers
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Language Model**: `google/flan-t5-base`
- **Vector DB**: Chroma
- **PDF Parsing**: PyPDF2

---

## 📂 How It Works

1. **PDF Upload**  
   Users upload one or more PDF files through the Streamlit UI.

2. **Text Extraction**  
   `PyPDF2` extracts text from all pages.

3. **Chunking**  
   Text is split into manageable chunks using `CharacterTextSplitter`.

4. **Embedding & Vector Storage**  
   Each chunk is converted into vector embeddings using HuggingFace and stored in Chroma DB.

5. **User Q&A**  
   Questions are processed, relevant chunks are retrieved, and a HuggingFace model (`flan-t5-base`) generates contextual answers.



