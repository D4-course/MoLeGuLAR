import requests
import streamlit as st

# interact with FastAPI endpoint
backend = "http://localhost:8000/smiles"

def process(url):
    send_data = {"file_name": "/scratch/arihanth.srikar/MoleGuLAR/Optimizer/temp_smiles.txt"}

    r = requests.post(url, json=send_data)
    return r.json()

my_dict = process(backend)
st.write(my_dict)
