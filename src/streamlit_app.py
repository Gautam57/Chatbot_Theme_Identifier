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


os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disable parallelism to avoid warnings
# os.environ["STREAMLIT_WATCHER_TYPE"] = "none" # Disable Streamlit watcher to avoid reloading issues
# Create upload folder if not exists
UPLOAD_DIR = "./uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)
chroma_db_path = "./vectorstore/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

with st.sidebar:
    st.header("📂 Document Manager")
    # 🔼 Upload new files
    uploaded = st.file_uploader("Upload Files", type=["txt", "pdf"], accept_multiple_files=True,key=st.session_state.uploader_key)
    if uploaded:
        st.session_state.file_uploaded = 0
        st.session_state.duplicate_files = []
        st.session_state.uploader_key += 1
        for file in uploaded:
            save_path = os.path.join(UPLOAD_DIR, file.name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(file.read())
                st.session_state.file_uploaded += 1
            else:
                st.session_state.duplicate_files.append(file.name)

        # ✅ Success message for uploaded files
        if st.session_state.file_uploaded > 0:
            st.success(f"✅ Uploaded {st.session_state.file_uploaded} new file(s).")

        # ⚠️ Show duplicates in clean table
        if st.session_state.duplicate_files:
            with st.expander("Duplicate Files", expanded=True):
                st.warning("⚠️ The following file(s) already exist. Please rename and re-upload:")
                st.table(pd.DataFrame({"Duplicate Files": st.session_state.duplicate_files}))
    Check_Available = st.button("Check Available Files")
    if Check_Available:
        # 📄 Get list of stored files
        st.session_state.file_list = [
            f for f in os.listdir(UPLOAD_DIR)
            if os.path.isfile(os.path.join(UPLOAD_DIR, f)) and f.endswith(('.txt', '.pdf',))
        ]

    if st.session_state.file_list:
        st.markdown("### 🗂️ Available Files")

        # Multi-select for embedding
        df = pd.DataFrame({"File": st.session_state.file_list})

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_selection("multiple", use_checkbox=True)
        grid_options = gb.build()

        selected = AgGrid(df, gridOptions=grid_options, update_mode='SELECTION_CHANGED')
        selected_files = selected["selected_rows"]
        if selected_files is not None:
            st.write(selected_files)
            selected_paths = [os.path.join(UPLOAD_DIR, row) for row in selected_files['File']]

        Delete_file = st.selectbox("🗑️ Delete a file:", st.session_state.file_list)
        if st.button(f"Delete") and Delete_file:
                os.remove(os.path.join(UPLOAD_DIR, Delete_file))
                # st.session_state.file_list.remove(Delete_file)
                st.session_state.file_list = [
                f for f in os.listdir(UPLOAD_DIR)
                if os.path.isfile(os.path.join(UPLOAD_DIR, f)) and f.endswith(('.txt', '.pdf',))
            ]
                st.warning(f"Deleted `{Delete_file}`. Refresh to update.")
                    # st.rerun()
    else:
        st.warning("No files available. Please upload files to proceed.")
        
    Create_Embeddings = st.button("Create Embeddings")
    if Create_Embeddings:
        

        # Set up embedding model (you can change model name if needed)
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if os.path.exists(chroma_db_path):
            st.session_state.vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)
            
            
            # Get already embedded filenames
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

        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        # List to hold all LangChain Document objects
        documents = []

        if st.session_state.sources_to_remove:
                # Remove documents whose source is not in selected_paths
                st.write(f"🗑️ Removing sources not in selection: {st.session_state.sources_to_remove}")
                # Chroma's API does not have a direct remove-by-metadata, so you may need to recreate the DB:
                all_docs = st.session_state.vectorstore.similarity_search("", k=10000)
                keep_docs = [doc for doc in all_docs if doc.metadata["source"] in selected_sources]
                # Recreate the vectorstore with only the kept docs
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=keep_docs,
                    embedding=embedding_model,
                    persist_directory=chroma_db_path
                )

        # Process all PDFs
        for filename in selected_paths :
            if filename.lower().endswith(".pdf") and filename not in existing_sources:
                pdf_path = filename
                print(f"Processing: {filename}")

                try:
                    # Convert PDF to images
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

                                # Clean chunk text to form paragraph
                                clean_text = ' '.join(chunk.page_content.splitlines())
                                clean_text = ' '.join(clean_text.split())

                                # Copy metadata and add paragraph_number
                                updated_metadata = dict(chunk.metadata)
                                updated_metadata["paragraph_number"] = paragraph_counter

                                # Create new Document
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
                chromeDB_Created = True
            else:
                st.session_state.vectorstore.add_documents(documents)
            print("✅ Chroma DB updated and saved.")
        else:
            print("ℹ️ No new PDFs to process.")
        
st.title("Theme Identifier")
# st.write("Find the theme of a document using AI.")
st.caption("This application allows you to upload documents and identify their themes using AI. "
           "You can also delete files from the system.")

query = st.text_input("Enter the theme you want to identify:")
if st.button("Identify Theme") and query:

    llm = ChatGroq(
        model_name="Llama3-8b-8192",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")  # Loads from environment
    )
    # query = "Things to know about Rosewood Trees Act ?"
    results = st.session_state.vectorstore.similarity_search_with_score(query, k=3)


    results_dt = pd.DataFrame(columns=["File Name", "Extracted Answer", "Citation", "Score"])

    for doc, score in results:
        meta = doc.metadata
        # print(f"\n🔹 Match {i+1}  (File: {meta['source']}  Page {meta['page_number']}/{meta['total_pages']}  ¶{meta['paragraph_number']})")
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

        # print("🧠 Groq answer for this paragraph:\n")
        # print(response.content)
        # print("────────────────────────────────────────────────────────")

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
    # Display the results in a table
    # st.dataframe(results_dt)
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
    # Show similar documents that were used for context (from the retrieval)
    st.subheader("🔍 Top Relevant Chunks")
    similar_docs = st.session_state.vectorstore.similarity_search_with_score(query, k=3)  # or .invoke(query) if that's your method
    with st.expander("Document Similarity Search"):
        for doc,score in similar_docs:
            st.write(doc.page_content, doc.metadata)
            st.write("--------------------------------")
    results_dt.to_csv("extracted_answers.csv", index=False)
    print("✅ Results saved to extracted_answers.csv")