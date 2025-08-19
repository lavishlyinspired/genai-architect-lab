import streamlit as st
from QApdf.data_ingestion import load_data
from QApdf.embedding import load_embedding
from QApdf.model_api import load_model
from llama_index.core import Document
import os
def main():
    st.set_page_config("QA with Documents")
    
   # doc=st.file_uploader("upload your document")
    
    uploaded_file = st.file_uploader("Upload a file")
        
    if uploaded_file is not None:
        # Save file locally
        save_path = os.path.join("data", uploaded_file.name)
        os.makedirs("data", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File saved to {save_path}")
    
       

    st.header("QA with Documents(Information Retrieval)")
    
    user_question= st.text_input("Ask your question")
    
    if st.button("submit & process"):
        
        with st.spinner("Processing..."):
            document=load_data()
            model=load_model()            
            query_engine=load_embedding(model,document)
            response = query_engine.query(user_question)
            st.write(response.response)
                
                
if __name__=="__main__":
    main()          