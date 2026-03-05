import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import tempfile

load_dotenv()

# ---------------- LLM ----------------

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.3
)

model = ChatHuggingFace(llm=llm)

# ---------------- Prompt ----------------

prompt = PromptTemplate(
    input_variables=["context"],
    template="""
You are a medical AI assistant.

Analyze the following blood report carefully.

Blood Report Data:
{context}

Give the result in **simple English**, easy to understand for a normal person.

Give the result in this format:

Possible Disease:
- What disease or health problem may exist

Confidence Level:
- Percentage of confidence

Suggested Medicines:
- Common medicines that doctors usually prescribe

Precautions:
- Diet or lifestyle advice

Medical Disclaimer:
- Tell the user this is AI advice and they must consult a doctor before taking any medicine.
"""
)

parser = StrOutputParser()

# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="Blood Report Analyzer", page_icon="🩸")

st.title("🩸 AI Blood Report Analyzer")
st.write("Upload your blood report and the AI will analyze it.")

uploaded_file = st.file_uploader("Upload Blood Report PDF", type="pdf")

if uploaded_file:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Combine all chunks as context
    context = "\n".join([doc.page_content for doc in docs])

    st.success("Blood report uploaded successfully!")

    if st.button("Analyze Blood Report"):

        with st.spinner("Analyzing report..."):

            chain = RunnableSequence(prompt, model, parser)

            result = chain.invoke({
                "context": context
            })

            st.subheader("🧠 AI Health Analysis")

            st.write(result)