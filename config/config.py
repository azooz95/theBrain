import os 
from dotenv import load_dotenv

from pathlib import Path
from dataclasses import dataclass

from pydantic import BaseModel

load_dotenv()

@dataclass
class FilePaths():
    script_path: str = Path(__file__).resolve()
    model_folder: str = script_path.parents[1] / 'src/assets/models/'
    paper_seg_model : str = script_path.parents[1] / model_folder / 'similified_model_paper_seg.onnx'
    data_base_path: str = script_path.parents[1] / 'src/assets/data/'
    milvus_db_path: str = script_path.parents[1] / 'db/milvus_demo.db/'
    google_credentials_path: str = script_path.parents[1] / 'src/assets/google workspace/credentials.json'
    google_token_path: str = script_path.parents[1] / 'src/assets/google workspace'
    microsoft_token_dir: str = script_path.parents[1] / 'src/assets/microsoft_tokens'
    microsoft_flow_dir: str = script_path.parents[1] / 'src/assets/microsoft_flows'

    def __post_init__(self):
        self.script_path = str(self.script_path)
        self.paper_seg_model = str(self.paper_seg_model)
        self.data_base_path = str(self.data_base_path)
        self.milvus_db_path = str(self.milvus_db_path)
        self.google_credentials_path = str(self.google_credentials_path)
        self.google_token_path = str(self.google_token_path)
        self.microsoft_token_dir = str(self.microsoft_token_dir)
        self.microsoft_flow_dir = str(self.microsoft_flow_dir)


@dataclass
class Ports():
    google_local_server: int = 3000
     
file_paths = FilePaths()



if __name__ == "__main__":

    print(Ports.google_local_server)