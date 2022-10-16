from fastapi import FastAPI
from pydantic import BaseModel

import streamlit as st
import subprocess

app = FastAPI()

class FileName(BaseModel):
    file_name: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/smiles")
async def update_item(f: FileName):
    print(f.file_name)
    smiles = []
    with open(f.file_name, "r") as f:
        smiles = f.readlines()
    smiles = [smile.split()[0] for smile in smiles]
    
    return {"smiles": smiles}



#cmd = ["python3", "model_logP_QED_switch.py", "--num_iterations", "1"]
# subprocess.Popen(cmd).wait()
# print(f"\n\n\n\nDONEEE\n\n\n\n")
#smiles = []

#with open("temp_smiles.txt", "r") as f:
#    smiles = f.readlines()

#smiles = [smile.split()[0] for smile in smiles]

#for i, smile in enumerate(smiles):
#    print(smile)

# st.write(smiles)
