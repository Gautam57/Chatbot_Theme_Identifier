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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
UPLOAD_DIR = "docs/"
os.makedirs(UPLOAD_DIR, exist_ok=True)
chroma_db_path = "./backend/data/chromaDB"

file_uploaded = 0
duplicate_files = []

with st.sidebar:
    st.header("📂 Document Manager")

    # 🔼 Upload new files
    uploaded = st.file_uploader("Upload Files", type=["txt", "pdf"], accept_multiple_files=True)
    if uploaded:
        for file in uploaded:
            save_path = os.path.join(UPLOAD_DIR, file.name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(file.read())
                file_uploaded += 1
            else:
                duplicate_files.append(file.name)

        # ✅ Success message for uploaded files
        if file_uploaded > 0:
            st.success(f"✅ Uploaded {file_uploaded} new file(s).")

        # ⚠️ Show duplicates in clean table
        if duplicate_files:
            st.warning("⚠️ The following file(s) already exist. Please rename and re-upload:")
            st.table(pd.DataFrame({"Duplicate Files": duplicate_files}))

        # 📄 Get list of stored files (all supported types)
        file_list = [
            f for f in os.listdir(UPLOAD_DIR)
            if os.path.isfile(os.path.join(UPLOAD_DIR, f)) and f.endswith(('.txt', '.md', '.pdf'))
        ]

        if file_list:
            st.markdown("### 🗂️ Available Files")
            df_files = pd.DataFrame({"File": file_list})

            gb = GridOptionsBuilder.from_dataframe(df_files)
            gb.configure_selection("multiple", use_checkbox=True)
            grid_options = gb.build()

            selected = AgGrid(df_files, gridOptions=grid_options, update_mode='SELECTION_CHANGED')
            selected_files = selected["selected_rows"]
            selected_paths = []
            if selected_files is not None:
                st.write(selected_files)
                selected_paths = [os.path.join(UPLOAD_DIR, row) for row in selected_files['File']]

            Delete_file = st.selectbox("🗑️ Delete a file:", file_list)
            if st.button("Delete"):
                os.remove(os.path.join(UPLOAD_DIR, Delete_file))
                st.warning(f"Deleted `{Delete_file}`. Refresh to update.")
                st.rerun()

            Create_Embeddings = st.button("Create Embeddings")
            if Create_Embeddings and selected_paths:
                embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                if os.path.exists(chroma_db_path):
                    st.session_state.vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)
                    existing_sources = set(
                        doc.metadata["source"] for doc in st.session_state.vectorstore.similarity_search("", k=500)
                    )
                    st.write(f"🗂️ Already embedded files: {existing_sources}")
                else:
                    st.session_state.vectorstore = None
                    existing_sources = set()
                    st.write("🆕 No existing Chroma DB found. Will create one.")

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                documents = []

                for filename in selected_paths:
                    st.write(filename)
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
                                    continue  # Skip blank pages

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

                if documents:
                    print(f"🔄 Adding {len(documents)} new chunks to Chroma DB...")
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=documents,
                            embedding=embedding_model,
                            persist_directory=chroma_db_path
                        )
                    else:
                        st.session_state.vectorstore.add_documents(documents)
                    print("✅ Chroma DB updated and saved.")
                else:
                    print("ℹ️ No new PDFs to process.")

# ------------------- MAIN APP -------------------

st.title("Theme Identifier")
st.caption("This application allows you to upload documents and identify their themes using AI. "
           "You can also delete files from the system.")

query = st.text_input("Enter the theme you want to identify:")
if st.button("Identify Theme") and query:
    llm = ChatGroq(
        model_name="Llama3-8b-8192",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
    results = st.session_state.vectorstore.similarity_search(query, k=5)
    results_dt = pd.DataFrame(columns=["File Name", "Extracted Answer", "Citation"])

    for i, doc in enumerate(results):
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
            "File Name": meta["source"],
            "Extracted Answer": answer,
            "Citation": f"Page {meta['page_number']}/{meta['total_pages']}  ,Para {meta['paragraph_number']}"
        }])
        results_dt = pd.concat([results_dt, new_row], ignore_index=True)

    st.dataframe(
        results_dt,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Extracted Answer": st.column_config.TextColumn("Extracted Answer", width="large"),
            "Citation": st.column_config.TextColumn("Citation", width="medium"),
            "File Name": st.column_config.TextColumn("File Name", width="medium"),
        }
    )

    st.subheader("🔍 Top Relevant Chunks")
    similar_docs = st.session_state.vectorstore.similarity_search(query, k=5)
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(similar_docs):
            st.write(doc.page_content, doc.metadata)
            st.write("--------------------------------")
    results_dt.to_csv("extracted_answers.csv", index=False)
    print("✅ Results saved to extracted_answers.csv")