# Document Theme Identifier
A Streamlit application that helps identify themes in documents using AI-powered text analysis and retrieval.

## Features

- **Document Management**:
  - Upload PDF and TXT files
  - View available documents
  - Delete unwanted files
- **AI-Powered Analysis**:
  - Create embeddings using HuggingFace models
  - Store documents in ChromaDB vector store
  - Query documents using Groq's Llama3 model
- **Theme Identification**:
  - Search for specific themes across documents
  - View extracted answers with citations
  - See relevant document chunks

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/document-theme-identifier.git
   cd document-theme-identifier
   ```

2. Create a virtual environment:
   ```bash
   conda create -p venv python==3.11.7 --y 
   source conda activate /venv
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR (required for PDF processing):
   - On Ubuntu/Debian:
     ```bash
     sudo apt install tesseract-ocr
     ```
   - On macOS:
     ```bash
     brew install tesseract
     ```
   - On Windows: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. Set up environment variables:
   Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Use the application:
   - Upload documents via the sidebar
   - Create embeddings for selected files
   - Enter a query to identify themes
   - View results with citations

## Configuration

The application can be configured by modifying the following constants in the code:

- `UPLOAD_DIR`: Directory for storing uploaded files (default: `./uploads/`)
- `chroma_db_path`: Directory for ChromaDB vector store (default: `./vectorstore/`)
- `embedding_model`: HuggingFace model name (default: `"all-MiniLM-L6-v2"`)
- `llm_model`: Groq model name (default: `"Llama3-8b-8192"`)

## Technical Details

### Document Processing Pipeline

1. **File Upload**:
   - Accepts PDF and TXT files
   - Stores files in the upload directory
   - Checks for duplicates

2. **Text Extraction**:
   - For PDFs: Uses `pdf2image` and `pytesseract` for OCR
   - For TXT files: Directly reads the content

3. **Text Chunking**:
   - Splits documents using `RecursiveCharacterTextSplitter`
   - Chunk size: 500 characters with 100-character overlap

4. **Embedding Generation**:
   - Uses HuggingFace's `all-MiniLM-L6-v2` model
   - Stores embeddings in ChromaDB

5. **Query Processing**:
   - Performs similarity search on the vector store
   - Uses Groq's Llama3 model for answer extraction
   - Returns answers with document citations

### Dependencies

- Core:
  - `streamlit` - Web application framework
  - `langchain` - AI orchestration framework
  - `langchain-huggingface` - HuggingFace integrations
  - `langchain-chroma` - ChromaDB integrations
  - `langchain-groq` - Groq API integration

- Document Processing:
  - `pdf2image` - PDF to image conversion
  - `pytesseract` - OCR for text extraction
  - `pandas` - Data handling and display


## Troubleshooting

1. **Tesseract OCR Errors**:
   - Ensure Tesseract is installed and in your PATH
   - For language-specific OCR, install appropriate language packs

2. **Groq API Errors**:
   - Verify your API key is set in the `.env` file
   - Check your internet connection

3. **Memory Issues**:
   - Reduce chunk size if processing large documents
   - Consider using smaller embedding models

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

Distributed under the MIT License. See `LICENSE` for more information.

