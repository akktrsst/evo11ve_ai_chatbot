import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
try:
    from dotenv import load_dotenv
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
except:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    

# Function to get text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {e}")
    return text


# Function to split text into chunks for processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return text_splitter.split_text(text)


# Function to create and save a FAISS vector store
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Function to update the conversation history
def update_conversation_history(user_question, assistant_response, max_turns=5):
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    st.session_state.conversation_history.append({"user": user_question, "assistant": assistant_response})
    if len(st.session_state.conversation_history) > max_turns:
        st.session_state.conversation_history.pop(0)


# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    You are a helpful assistant strictly answering questions only from the provided NCERT/CBSE curriculum chapter.

    Rules:
    1. Only answer questions related to the provided chapter content.
    2. If a question is outside the curriculum, respond with: "Your curriculum does not have the information about [topic]. Please ask questions within the curriculum."
    3. Use conversation history to relate to previous answers.

    'few_shot_examples' : 
        User: What is photosynthesis?
        Assistant: Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water.

        User: Tell me more about it.
        Assistant: Photosynthesis involves the conversion of light energy into chemical energy, which is stored in glucose molecules.

        User: What is quantum mechanics?
        Assistant: Your curriculum does not have the information about quantum mechanics. Please ask questions within the curriculum.

        User: How does photosynthesis relate to respiration?
        Assistant: Photosynthesis produces oxygen and glucose, which are used in respiration to release energy for the cell's activities.

    ### Chapter Content:
    {context}

    ### Conversation History:
    {history}

    ### Current User Query:
    {user_question}

    ### Response:
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=openai_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "history", "user_question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Function to handle user queries
def user_input(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(query=user_question, k=5)

    # Build conversation history string
    history = "\n".join([f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in st.session_state.get('conversation_history', [])])
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "history": history, "user_question": user_question}, return_only_outputs=True)
    assistant_response = response["output_text"]
    
    update_conversation_history(user_question, assistant_response)
    
    return assistant_response


# Main Streamlit app
def main():
    st.set_page_config(page_title="NCERT/CBSE Book Chatbot", page_icon="📚")
    st.title("📚 NCERT/CBSE Book Chatbot 🤖")

    # Sidebar for PDF upload
    with st.sidebar:
        st.image("img/Idea.jpg")
        st.subheader("Upload Your Books")
        pdf_docs = st.file_uploader("Upload in PDF format", accept_multiple_files=True)

        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDFs processed successfully!")
                    else:
                        st.error("No valid text extracted from the uploaded PDFs.")
            else:
                st.error("Please upload at least one PDF.")

    # Display conversation history if available
    if 'conversation_history' in st.session_state:
        for turn in st.session_state.conversation_history:
            st.markdown(f"<div style='color:black;'>User: {turn['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:blue;'>Assistant: {turn['assistant']}</div>", unsafe_allow_html=True)

    # User question input
    user_questions = st.text_area("Ask multiple questions (separate with newlines):")
    if user_questions:
        # Split the user's input into separate questions
        questions = user_questions.split("\n")
        
        for question in questions:
            if question.strip():  # Only process non-empty questions
                assistant_response = user_input(question.strip())
                st.markdown(f"<div style='color:blue;'>Assistant: {assistant_response}</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center; color: white;">
            © 2024 Abhishek Kumar | EVO11VE AI | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
