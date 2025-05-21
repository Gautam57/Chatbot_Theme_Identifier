import os
import pytesseract
import pandas as pd

from pdf2image import convert_from_path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


# Disable tokenizer parallelism to avoid console warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create upload and vectorstore directories if they don't exist
UPLOAD_DIR = "./uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)
chroma_db_path = "./vectorstore/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state variables if not already set
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = 0
if "duplicate_files" not in st.session_state:
    st.session_state.duplicate_files = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "file_list" not in st.session_state:
    st.session_state.file_list = []
if "sources_to_remove" not in st.session_state:
    st.session_state.sources_to_remove = []

# Sidebar for document upload and management
with st.sidebar:
    st.header("📂 Document Manager")

    # File uploader allowing multiple PDF/TXT files
    uploaded = st.file_uploader("Upload Files", type=["txt", "pdf"], accept_multiple_files=True, key=st.session_state.uploader_key)
    
    if uploaded:
        # Reset counters and flags
        st.session_state.file_uploaded = 0
        st.session_state.duplicate_files = []
        st.session_state.uploader_key += 1
        
        # Save new files, skip duplicates
        for file in uploaded:
            save_path = os.path.join(UPLOAD_DIR, file.name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(file.read())
                st.session_state.file_uploaded += 1
            else:
                st.session_state.duplicate_files.append(file.name)

        # Notify about newly uploaded files
        if st.session_state.file_uploaded > 0:
            st.success(f"✅ Uploaded {st.session_state.file_uploaded} new file(s).")

        # Notify about duplicate files
        if st.session_state.duplicate_files:
            with st.expander("Duplicate Files", expanded=True):
                st.warning("⚠️ The following file(s) already exist. Please rename and re-upload:")
                st.table(pd.DataFrame({"Duplicate Files": st.session_state.duplicate_files}))

    # Check which files are already uploaded
    Check_Available = st.button("Check Available Files")
    if Check_Available:
        st.session_state.file_list = [
            f for f in os.listdir(UPLOAD_DIR)
            if os.path.isfile(os.path.join(UPLOAD_DIR, f)) and f.endswith(('.txt', '.pdf',))
        ]

    if st.session_state.file_list:
        # Display list of uploaded files and allow selection
        st.markdown("### 🗂️ Available Files")

        df = pd.DataFrame({"File": st.session_state.file_list})

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_selection("multiple", use_checkbox=True)
        grid_options = gb.build()

        selected = AgGrid(df, gridOptions=grid_options, update_mode='SELECTION_CHANGED')
        selected_files = selected["selected_rows"]
        if selected_files is not None:
            st.write(selected_files)
            selected_paths = [os.path.join(UPLOAD_DIR, row) for row in selected_files['File']]

        # File deletion section
        Delete_file = st.selectbox("🗑️ Delete a file:", st.session_state.file_list)
        if st.button(f"Delete") and Delete_file:
                os.remove(os.path.join(UPLOAD_DIR, Delete_file))
                st.session_state.file_list = [
                f for f in os.listdir(UPLOAD_DIR)
                if os.path.isfile(os.path.join(UPLOAD_DIR, f)) and f.endswith(('.txt', '.pdf',))
            ]
                st.warning(f"Deleted `{Delete_file}`. Refresh to update.")

    else:
        st.warning("No files available. Please upload files to proceed.")
        
    # Create Embeddings section
    Create_Embeddings = st.button("Create Embeddings")
    if Create_Embeddings:
        # Set up embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if os.path.exists(chroma_db_path):
            st.session_state.vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)
            
            # Retrieve embedded document metadata
            existing_sources = set(
                doc.metadata["source"] for doc in st.session_state.vectorstore.similarity_search("", k=10000)
            )
            selected_sources = set(selected_paths)
            st.session_state.sources_to_remove = existing_sources - selected_sources
            st.write(f"🗂️ Already embedded files: {existing_sources}")
        else:
            st.session_state.vectorstore = None
            existing_sources = set()
            st.write("🆕 No existing Chroma DB found. Will create one.")

        # Initialize text splitter for chunking documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        documents = []

        # Remove unselected files from vectorstore
        if st.session_state.sources_to_remove:
            st.write(f"🗑️ Removing sources not in selection: {st.session_state.sources_to_remove}")
            all_docs = st.session_state.vectorstore.similarity_search("", k=10000)
            keep_docs = [doc for doc in all_docs if doc.metadata["source"] in selected_sources]
            st.session_state.vectorstore = Chroma.from_documents(
                documents=keep_docs,
                embedding=embedding_model,
                persist_directory=chroma_db_path
            )

        # Process each selected PDF for OCR and chunking
        for filename in selected_paths :
            if filename.lower().endswith(".pdf") and filename not in existing_sources:
                pdf_path = filename
                print(f"Processing: {filename}")

                try:
                    pages = convert_from_path(pdf_path)
                    total_pages = len(pages)
                    print(f"Total pages: {total_pages}")

                    for page_num, page in enumerate(pages, start=1):
                        text = pytesseract.image_to_string(page)
                        text_length = len(text.strip())

                        if text_length == 0:
                            continue

                        metadata = {
                            "source": filename,
                            "page_number": page_num,
                            "total_pages": total_pages,
                            "text_length": text_length
                        }

                        doc = Document(page_content=text, metadata=metadata)
                        raw_chunks = text_splitter.split_documents([doc])
                        paragraph_counter = 0

                        for chunk in raw_chunks:
                            paragraph_counter += 1

                            clean_text = ' '.join(chunk.page_content.splitlines())
                            clean_text = ' '.join(clean_text.split())

                            updated_metadata = dict(chunk.metadata)
                            updated_metadata["paragraph_number"] = paragraph_counter

                            cleaned_chunk = Document(page_content=clean_text, metadata=updated_metadata)
                            documents.append(cleaned_chunk)

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        # Add new documents to vectorstore
        if documents:
            print(f"🔄 Adding {len(documents)} new chunks to Chroma DB...")
            
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embedding_model,
                    persist_directory=chroma_db_path
                )
                chromeDB_Created = True
            else:
                st.session_state.vectorstore.add_documents(documents)
            print("✅ Chroma DB updated and saved.")
        else:
            print("ℹ️ No new PDFs to process.")

# Title and user query input
st.title("Theme Identifier")
st.caption("This application allows you to upload documents and identify their themes using AI. "
           "You can also delete files from the system.")

query = st.text_input("Enter the theme you want to identify:")

# Trigger search and extraction using vectorstore and LLM
if st.button("Identify Theme") and query:
    # Initialize LLM from Groq API
    llm = ChatGroq(
        model_name="Llama3-8b-8192",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Perform semantic search
    results = st.session_state.vectorstore.similarity_search_with_score(query, k=3)

    results_dt = pd.DataFrame(columns=["File Name", "Extracted Answer", "Citation", "Score"])

    for doc, score in results:
        meta = doc.metadata
        system_prompt = (
            "You are an AI assistant. Use ONLY the given CONTEXT to answer the USER question. "
            "If the answer is not present in the context, reply: 'Not found in this paragraph.'."
        )

        human_prompt = (
            f"CONTEXT:\n{doc.page_content}\n\n"
            f"USER QUESTION:\n{query}"
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        answer = response.content

        new_row = pd.DataFrame([{
                "File Name":        meta["source"],
                "Extracted Answer": answer,
                "Citation":         f"Page {meta['page_number']}/{meta['total_pages']}  ,Para {meta['paragraph_number']}",
                "Score": score
            }])
        
        results_dt = pd.concat([results_dt, new_row], ignore_index=True)

    final_results = results_dt.sort_values(by="Score", ascending=False)
    final_results = final_results.drop(columns=["Score"])

    # Display the final answers
    st.dataframe(
        final_results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Extracted Answer": st.column_config.TextColumn("Extracted Answer", width="large"),
            "Citation": st.column_config.TextColumn("Citation", width="medium"),
            "File Name": st.column_config.TextColumn("File Name", width="medium"),
        }
    )

    # Show context used for answering
    st.subheader("🔍 Top Relevant Chunks")
    similar_docs = st.session_state.vectorstore.similarity_search_with_score(query, k=3)
    with st.expander("Document Similarity Search"):
        for doc, score in similar_docs:
            st.write(doc.page_content, doc.metadata)
            st.write("--------------------------------")

    results_dt.to_csv("extracted_answers.csv", index=False)
    st.success("✅ Results saved to extracted_answers.csv")