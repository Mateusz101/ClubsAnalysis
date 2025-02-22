import streamlit as st

def set_style(): # Funkcja ustawiająca styl dla strony
    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        
        h1 {
            font-size: 35px;
        }
        
        .title {
            background-color: #11E076;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 125px;
            border-radius: 15px; 
        }

        .subtitle {
            background-color: #78EB5C;
            width: 100%;
            display: flex; 
            align-items: center; 
            height: 75px;
            padding: 0 10px; 
            border-radius: 15px; 
            margin-bottom: 15px;
        }
        
       .container {
            display: flex; 
            justify-content: center; 
            align-items: center;  
            margin-bottom: 15px;
        }

        .subsubtitle {
            background-color: #ACEB55;
            width: 50%;
            display: flex; 
            align-items: center;  
            height: 50px;
            justify-content: center;
            border-radius: 15px;  
            margin-top:20px;
        }

        .table-container{
            display: flex; justify-content: center; align-items: center;
        }
        
        .table-content{
            max-height: 300px; overflow-y: auto; border-radius: 8px;
        }

        .subsubtitle h3 {
            color: #ffffff;
        }
        
        .subtitle h2 {
            color: #ffffff;
        }
            
        .title h1 {
            color: #ffffff;
        }
        
        
        </style>
        """,
        unsafe_allow_html=True
    )


