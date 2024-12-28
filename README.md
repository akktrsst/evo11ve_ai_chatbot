# 📚 NCERT/CBSE Book Chatbot 🤖

A Streamlit-based conversational chatbot designed to help students interact with NCERT/CBSE curriculum-based chapters. Upload PDF books, ask questions, and get precise answers directly from the uploaded chapter content. The chatbot ensures it stays within the context of the uploaded material and helps students focus on their syllabus.

---

## 🚀 Features

- **PDF Upload Support**: Upload NCERT/CBSE books or specific chapters in PDF format.
- **Text Extraction**: Extracts and processes text from uploaded PDFs.
- **Conversational Chatbot**: Engages in a conversation while keeping answers strictly within the curriculum.
- **Vector Search with FAISS**: Finds the most relevant text chunks to answer user questions.
- **Conversation History**: Retains a rolling window of conversation history for context-aware responses.
- **Curriculum-Specific Answers**: Rejects questions outside the provided syllabus.

---

## 📋 Requirements

- Python 3.9+
- OpenAI API Key
- Libraries:
  - `streamlit`
  - `PyPDF2`
  - `langchain`
  - `langchain_openai`
  - `faiss-cpu`
  - `python-dotenv`

---

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/akktrsst/evo11ve_ai_chatbot.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project directory.
   - Add the following line:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## 🖼️ Screenshots

### Home Page

![Home Page](https://github.com/akktrsst/evo11ve_ai_chatbot/blob/master/img/home.jpg)

### PDF Upload

![PDF Upload](https://github.com/akktrsst/evo11ve_ai_chatbot/blob/master/img/pdf_upload.jpg)

### Chatbot Interaction

![Chat Interaction](https://github.com/akktrsst/evo11ve_ai_chatbot/blob/master/img/chatbot.jpg)

---

## 📹 Demo Video

[![Demo Video](https://github.com/akktrsst/evo11ve_ai_chatbot/blob/master/img/home.jpg)](https://github.com/akktrsst/evo11ve_ai_chatbot/raw/master/img/demo.mp4)

---

## 🖥️ Streamlit App 

[Streamlit App Link](https://evo11veai.streamlit.app/)

## 📚 Download NCERT Book's Chapters
[NCERT Book Link](https://ncert.nic.in/textbook.php?)

## ⚙️ How It Works

1. **Upload PDFs**: Users upload NCERT books or chapters via the sidebar.
2. **Text Processing**: Extracts and splits text into manageable chunks using LangChain's text splitter.
3. **Vector Store**: Stores text chunks as vectors using FAISS for efficient similarity search.
4. **Conversational QA**:
   - Uses OpenAI's GPT-based model to respond to questions.
   - Searches for the most relevant chunks using vector similarity.
   - Follows a strict rule to stay within the curriculum.

---

## 📂 Project Structure

```
.
├── app.py               # Main Streamlit app
├── requirements.txt     # Dependencies
├── .env                 # Environment variables
├── img/                 # Images used in the app
├── faiss_index/         # Stored FAISS vector index
└── README.md            # Documentation
```

---

## 🤝 Contribution

Contributions are welcome! Feel free to:
- Report issues
- Suggest features
- Submit pull requests

---

## 🔒 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## 👨‍💻 Author

**Abhishek Kumar**  

---

### 🌟 If you like this project, don't forget to star it! 🌟

#### Follow me on [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abhishekrishav/) &nbsp; [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/akktrsst)