import streamlit as st
from query_engine import QueryEngine
from document_processor import DocumentProcessor
import os
import tempfile
from pathlib import Path
import docx
import PyPDF2
import pandas as pd
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Company Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)

# Enhanced document management
def show_document_manager():
    st.subheader("📋 Document Database")
    
    # Show uploaded documents
    if os.path.exists("uploads"):
        uploaded_files = os.listdir("uploads")
        if uploaded_files:
            st.write("**Uploaded Documents:**")
            for file in uploaded_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {file}")
                with col2:
                    if st.button(f"🗑️", key=f"delete_{file}"):
                        os.remove(f"uploads/{file}")
                        st.success(f"Deleted {file}")
                        st.experimental_rerun()

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {str(e)}")
        return None

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None

def extract_text_from_csv(file):
    """Extract text from CSV file"""
    try:
        # Read CSV
        df = pd.read_csv(file)
        
        # Convert to text format
        text = f"CSV Data from {file.name}:\n\n"
        text += f"Columns: {', '.join(df.columns)}\n\n"
        text += f"Total rows: {len(df)}\n\n"
        
        # Add first few rows as context
        text += "Sample data:\n"
        text += df.head(10).to_string(index=False)
        
        # Add summary statistics if numeric columns exist
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            text += "\n\nSummary Statistics:\n"
            text += df[numeric_cols].describe().to_string()
        
        return text
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return None

def process_uploaded_file(uploaded_file, doc_processor):
    """Process an uploaded file and add to vector store"""
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Get file extension
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        # Extract text based on file type
        text_content = None
        
        if file_extension == '.txt':
            # Text file
            text_content = str(uploaded_file.read(), "utf-8")
            
        elif file_extension == '.docx':
            # Word document
            text_content = extract_text_from_docx(uploaded_file)
            
        elif file_extension == '.pdf':
            # PDF document
            text_content = extract_text_from_pdf(uploaded_file)
            
        elif file_extension == '.csv':
            # CSV file
            text_content = extract_text_from_csv(uploaded_file)
            
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return False
        
        if not text_content or len(text_content.strip()) == 0:
            st.error("Could not extract text from the file or file is empty")
            return False
        
        # Save the extracted content as a text file
        file_path = f"uploads/{uploaded_file.name}_processed.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        # Process the file
        document_id = Path(uploaded_file.name).stem
        success = doc_processor.process_text_file(file_path, document_id)
        
        if success:
            st.success(f"✅ Successfully processed '{uploaded_file.name}'!")
            st.info(f"📄 Extracted {len(text_content)} characters of text")
            return True
        else:
            st.error("Failed to process the document")
            return False
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False

def main():
    st.title("🤖 Company Documents Q&A Assistant")
    st.write("Ask me anything about company policies and procedures!")
    
    # Initialize components
    if 'query_engine' not in st.session_state:
        with st.spinner("Initializing AI models..."):
            st.session_state.query_engine = QueryEngine()
    
    if 'doc_processor' not in st.session_state:
        with st.spinner("Setting up document processor..."):
            st.session_state.doc_processor = DocumentProcessor()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Sample documents section
        st.subheader("Load Sample Documents")
        if st.button("Load Sample Documents"):
            with st.spinner("Loading sample documents..."):
                st.session_state.doc_processor.add_sample_documents()
                st.success("Sample documents loaded!")
        
        st.write("---")
        
        # File upload section
        st.subheader("📤 Upload Your Documents")
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['txt', 'pdf', 'docx', 'csv'],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, DOCX, CSV"
        )
        
        if uploaded_files:
            st.write(f"**Selected files ({len(uploaded_files)}):**")
            file_type_icons = {
                ".pdf": "📑", ".txt": "📄", ".docx": "📝", ".csv": "📊"
            }
            for file in uploaded_files:
                ext = Path(file.name).suffix.lower()
                icon = file_type_icons.get(ext, "📁")
                st.write(f"{icon} {file.name} ({file.size} bytes)")

            
            if st.button("🚀 Process All Files"):
                progress_bar = st.progress(0)
                success_count = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    st.write(f"Processing: {uploaded_file.name}...")
                    if process_uploaded_file(uploaded_file, st.session_state.doc_processor):
                        success_count += 1
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.success(f"✅ Successfully processed {success_count}/{len(uploaded_files)} files!")
        
        st.write("---")
        
        # Vector store statistics
        st.subheader("📊 Database Stats")
        if st.button("🔄 Refresh Stats"):
            stats = st.session_state.query_engine.get_vector_store_stats()
            st.metric("Total Documents", stats['total_documents'])
            st.metric("Vector Dimension", stats['dimension'])
            # Show live count of uploaded docs
            st.metric("📂 Uploaded Docs", len(os.listdir("uploads")) if os.path.exists("uploads") else 0)

        
        # Clear database option
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.checkbox("I confirm I want to clear all documents"):
                st.session_state.doc_processor.clear_all_documents()
                st.success("All documents cleared!")
        
        st.write("---")
        
        # Model settings
        st.header("🤖 Model Settings")
        use_advanced = st.checkbox("Use Advanced AI Generation", help="Uses HuggingFace model for responses (experimental)")
        
        if use_advanced:
            st.info("⚡ Advanced mode uses local HuggingFace models")
        else:
            st.info("🚀 Simple mode uses template-based responses (faster)")
        
        st.write("---")
        st.write("**Available sample questions:**")
        st.write("- What's our vacation policy?")
        st.write("- How do I submit an expense report?")
        st.write("- Who should I contact for IT issues?")
        st.write("- How many vacation days do I get?")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        # Quick question buttons
        st.write("**Quick questions:**")
        col1_q, col2_q, col3_q = st.columns(3)
        
        with col1_q:
            if st.button("📋 Vacation Policy"):
                st.session_state.current_question = "What's our vacation policy?"
        
        with col2_q:
            if st.button("💰 Expense Reports"):
                st.session_state.current_question = "How do I submit expense reports?"
        
        with col3_q:
            if st.button("🔧 IT Support"):
                st.session_state.current_question = "Who should I contact for IT issues?"
        
        # Question input
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
        
        question = st.text_area(
            "Type your question here:",
            value=st.session_state.current_question,
            placeholder="e.g., What's our vacation policy?",
            height=100
        )

        # Clear the current question after displaying
        if st.session_state.current_question:
            st.session_state.current_question = ""
        
        col_ask, col_clear = st.columns([3, 1])
        
        with col_ask:
            ask_button = st.button("🔍 Ask Question", type="primary")
        
        with col_clear:
            if st.button("🧹 Clear Chat"):
                st.session_state.chat_history = []
                st.success("Chat cleared!")
        
        if ask_button and question:
            with st.spinner("Searching documents and generating answer..."):
                try:
                    # Get answer with selected mode
                    answer = st.session_state.query_engine.ask_question(question, use_advanced)
                    
                    # Initialize chat history if it doesn't exist
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': answer
                    })
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
        
        # Display chat history
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            st.header("💭 Chat History")
            for chat in reversed(st.session_state.chat_history):
                st.markdown(f"""
                <div style='background-color:#f4f4f4; padding:15px; border-radius:10px; margin-bottom:10px'>
                    <strong>👤 You:</strong> {chat['question']}  
                    <br/><strong>🤖 Assistant:</strong> {chat['answer']}
                </div>
                """, unsafe_allow_html=True)

    
    with col2:
        st.header("📋 Instructions")
        st.write("""
        **How to use this assistant:**
        
        1. **Upload Documents**: Use the sidebar to upload your company documents (PDF, DOCX, TXT, CSV)
        
        2. **Process Files**: Click "Process All Files" to add them to the knowledge base
        
        3. **Ask Questions**: Type your questions in natural language
        
        4. **Get Answers**: The AI will search through your documents and provide relevant answers
        
        **Supported File Types:**
        - 📄 **TXT**: Plain text files
        - 📑 **PDF**: PDF documents
        - 📝 **DOCX**: Word documents  
        - 📊 **CSV**: Spreadsheet data
        
        **Tips for Better Results:**
        - Be specific in your questions
        - Use keywords from your documents
        - Try different phrasings if needed
        """)
        
        st.write("---")
        st.write("**Example Questions:**")
        st.code("""
- What is the company's remote work policy?
- How do I request time off?
- What are the expense reimbursement rules?
- Who is my HR contact?
- What benefits does the company offer?
- How do I access the VPN?
        """)

if __name__ == "__main__":
    main()