import streamlit as st
import tempfile
from text_extraction import *
from load_model import *

st.set_page_config(page_title="AI Text Summarizer", layout="wide")

if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'summarized_text' not in st.session_state:
    st.session_state.summarized_text = ""
if 'preprocessed_text' not in st.session_state:
    st.session_state.preprocessed_text = ""
if 'extract_clicked' not in st.session_state:
    st.session_state.extract_clicked = False
if 'summarize_clicked' not in st.session_state:
    st.session_state.summarize_clicked = False
if 'is_extracted' not in st.session_state:
    st.session_state.is_extracted = False

def on_extract_click():
    st.session_state.extract_clicked = True
    st.session_state.summarize_clicked = False

def on_summarize_click():
    st.session_state.summarize_clicked = True

def main():
    with st.sidebar:
        st.image("resources/logo.png")
        st.markdown("---")
        uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx", "txt"])
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; margin-top: -10px;">
            <p style="margin-bottom: 15px; font-size: 15px;">Created by <b>Sneha Kondapalli</b></p>
            <div style="display: flex; justify-content: center; gap: 20px;">
                <a href="https://www.linkedin.com/in/sneh-kondapall/" target="_blank">
                    <img src="https://img.icons8.com/?size=40&id=8808&format=png&color=228BE6" alt="LinkedIn" style="background-color: white;"/>
                </a>
                <a href="https://github.com/snehakondapalli" target="_blank"> 
                    <img src="https://img.icons8.com/?size=40&id=106564&format=png&color=000000" alt="GitHub" style="background-color: white; border-radius: 50%;"/>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    selected_model = st.radio(
        "Select Model to Summarize Text",
        ["llama-3.3-70b-versatile", "google/pegasus-xsum"],
        horizontal=True
    )

    if selected_model == "google/pegasus-xsum":
        col1_1, col1_2, col1_3 = st.columns([1, 1, 0.5], vertical_alignment="center")
        with col1_1:
            min_length = st.slider("Min Length", 10, 500, 100, step=10)
        with col1_2:
            max_length = st.slider("Max Length", 50, 4000, 1500, step=10)
        with col1_3:
            do_sample = st.checkbox("Enable Sampling (do_sample)", value=False)
    elif selected_model == "llama-3.3-70b-versatile":
        col2_1, col2_2, col2_3 = st.columns([1, 1, 1], vertical_alignment="center")
        with col2_1:
            max_tokens = st.slider("Max Tokens", min_value=50, max_value=4000, value=750, step=10)
        with col2_2:
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.05)
        with col2_3:
            top_p = st.slider("Top-p (nucleus sampling)", min_value=0.1, max_value=1.5, value=1.0, step=0.05)


    col1, vl, col2 = st.columns([0.49, 0.02, 0.49], gap="medium")
    
    with col1:
        st.subheader("Extracted Text")
        st.button("Extract Text", on_click=on_extract_click)
        if st.session_state.extract_clicked:
            if uploaded_file is None:
                st.warning("Please upload a file.")
            else:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                # Extract text
                with st.spinner("Extracting text..."):
                    if file_extension == 'pdf':
                        extracted_text = get_text_from_pdf(temp_path)
                    elif file_extension == 'docx':
                        extracted_text = get_text_from_docx(temp_path)
                    elif file_extension == 'txt':
                        extracted_text = get_text_from_txt(temp_path)
                    else:
                        st.warning("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
                        extracted_text = ""
                    preprocessed_text = preprocess_text(extracted_text) if extracted_text else ""
                    st.session_state.extracted_text = extracted_text
                    st.session_state.preprocessed_text = preprocessed_text
                    st.write("Extracted text length:", len(extracted_text))
                    st.session_state.is_extracted = bool(extracted_text)
                    if st.session_state.is_extracted:
                        with st.expander("Extracted Text", expanded=True):
                            st.markdown(f"""
                            <div style="padding: 15px; border: 1px solid white; border-radius: 10px; margin-bottom: 20px; height: 250px; overflow-y: scroll">{st.session_state.extracted_text}</div>
                            """, unsafe_allow_html=True)
                            st.download_button("Download Extract", st.session_state.extracted_text, file_name=f"text_extract_{uploaded_file.name.split('.')[0]}.txt")
            
    with vl:
        st.markdown("""
        <div style="background-color: white; width: 2px; padding-top: 500px;"></div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Summarized Text")
        st.button("Summarize Text", on_click=on_summarize_click)
        warning_placeholder = st.empty()
        warning_placeholder2 = st.empty()
        if st.session_state.summarize_clicked:
            if not st.session_state.is_extracted:
                warning_placeholder.warning("Please extract text first.")
            else:
                with st.spinner("Loading model {}...".format(selected_model)):
                    try:
                        if selected_model == "google/pegasus-xsum":
                            summarizer = Summarizer_HuggingFace("google/pegasus-xsum")
                        elif selected_model == "llama-3.3-70b-versatile":
                            api_key = st.secrets["GROQ_API_KEY"]
                            summarizer = Summarizer_Groq("llama-3.3-70b-versatile", api_key)
                    except Exception as e:
                        warning_placeholder.markdown(
                            f"""
                            <div style="background-color: #3e3b16; color: #ffffc2; padding: 15px; border-radius: 10px;">
                                <strong>Error loading model:</strong> <br>{e}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        summarizer = None
                with st.spinner("Summarizing..."):
                    if summarizer:
                        if selected_model == "google/pegasus-xsum":
                            chunks = summarizer._chunk_texts(st.session_state.preprocessed_text)
                            if len(chunks) > 1:
                                warning_placeholder.warning(f"Text is too long. It has been chunked into {len(chunks)} chunks. This may take a while...")
                            try:
                                summary = summarizer(
                                    chunks,
                                    min_length=min_length,
                                    max_length=max_length,
                                    do_sample=do_sample
                                )
                            except Exception as e:
                                warning_placeholder2.markdown(
                                    f"""
                                    <div style="background-color: #3e3b16; color: #ffffc2; padding: 15px; border-radius: 10px;">
                                        <strong>Error loading model:</strong> <br>{e}
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                        elif selected_model == "llama-3.3-70b-versatile":
                            tokens_length = summarizer._count_tokens(st.session_state.preprocessed_text)
                            if tokens_length > summarizer.max_input_tokens:
                                warning_placeholder.warning(f"Text is too long. Only the first {summarizer.max_input_tokens} tokens will be used. This may take a while...")
                            try:
                                summary = summarizer(
                                    st.session_state.preprocessed_text,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    top_p=top_p
                                )
                            except Exception as e:
                                warning_placeholder2.markdown(
                                    f"""
                                    <div style="background-color: #3e3b16; color: #ffffc2; padding: 15px; border-radius: 10px;">
                                        <strong>Error loading model:</strong> <br>{e}
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                    if summary:
                        st.session_state.summarized_text = summary
                        st.write("Summarized text length:", len(summary))
                        if st.session_state.is_extracted:
                            with st.expander("Summarized Text", expanded=True):
                                st.markdown(f"""
                                <div style="padding: 15px; border: 1px solid white; border-radius: 10px; margin-bottom: 20px; height: 250px; overflow-y: scroll">{st.session_state.summarized_text}</div>
                                """, unsafe_allow_html=True)
                                st.download_button("Download Summary", st.session_state.summarized_text, file_name=f"summary_{uploaded_file.name.split('.')[0]}.txt")

if __name__ == "__main__":
    main()