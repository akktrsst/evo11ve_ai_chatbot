import pandas as pd
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
            return e
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

    ### Current User Query:
    {user_question}

    ### Response:
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=openai_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Function to handle user queries
def user_input(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(query=user_question, k=5)

    # Build conversation string
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "user_question": user_question}, return_only_outputs=True)
    assistant_response = response["output_text"]
    
    return assistant_response

# Questions for testing
in_context_questions = [
    "What is the focal length of a convex lens with a radius of curvature of 20 cm?",
    "Define the principal focus of a concave mirror.",
    "Why do convex mirrors provide a wider field of view?",
    "How does a concave mirror form a real image?",
    "What is the power of a lens with a focal length of 50 cm?",
    "Explain the relationship between the radius of curvature and the focal length of a mirror.",
    "What are the two laws of reflection?",
    "How is the refractive index related to the speed of light?",
    "What happens to light when it moves from air to water?",
    "List some uses of convex lenses."
]

out_of_context_questions = [
    "Who discovered the laws of reflection?",
    "What is quantum theory?",
    "How does a telescope work?",
    "What is the speed of light in a vacuum in miles per second?",
    "Can a plane mirror form a magnified image?",
    "What is the history of concave mirrors?",
    "How does diffraction affect light?",
    "What are the applications of fiber optics?",
    "What is the cost of a convex mirror?",
    "How do prisms split light into colors?"
]

# Function to test chatbot
def test_chatbot(questions, expected_contextual):
    results = []
    for question in questions:
        response = user_input(question)  
        if "Your curriculum does not have the information" in response:
            is_contextual = False
        else:
            is_contextual = True
        
        results.append({
            "Question": question,
            "Response": response,
            "Expected Contextual": expected_contextual,
            "Is Contextual": is_contextual
        })
    return results

# Test the chatbot
in_context_results = test_chatbot(in_context_questions, True)
out_of_context_results = test_chatbot(out_of_context_questions, False)

# Combine results
test_results = in_context_results + out_of_context_results

# Create a DataFrame for analysis
df = pd.DataFrame(test_results)

# Calculate confusion matrix
tp = len(df[(df["Expected Contextual"] == True) & (df["Is Contextual"] == True)])
tn = len(df[(df["Expected Contextual"] == False) & (df["Is Contextual"] == False)])
fp = len(df[(df["Expected Contextual"] == False) & (df["Is Contextual"] == True)])
fn = len(df[(df["Expected Contextual"] == True) & (df["Is Contextual"] == False)])

confusion_matrix = {
    "True Positives (TP)": tp,
    "True Negatives (TN)": tn,
    "False Positives (FP)": fp,
    "False Negatives (FN)": fn
}

# Display results
print("Test Results:")
print(df)
df.to_csv("result.csv")
print("\nConfusion Matrix:")
for key, value in confusion_matrix.items():
    print(f"{key}: {value}")
