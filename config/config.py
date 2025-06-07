from pathlib import Path
from dataclasses import dataclass
import os 

@dataclass
class FilePaths():
    script_path: str = Path(__file__).resolve()
    model_folder: str = script_path.parents[1] / 'src/assests/models/'
    paper_seg_model : str = script_path.parents[1] / model_folder / 'similified_model_paper_seg.onnx'
    data_base_path: str = script_path.parents[1] / 'src/assests/data/'
    milvus_db_path: str = script_path.parents[1] / 'db/milvus_demo.db/'
    google_crenditials_path: str = script_path.parents[1] / 'src/assets/google workspaces/credentials.json'
    google_token_path: str = script_path.parents[1] / 'src/assets/google workspace/token.json'

    def __post_init__(self):
        self.script_path = str(self.script_path)
        self.paper_seg_model = str(self.paper_seg_model)
        self.data_base_path = str(self.data_base_path)
        self.milvus_db_path = str(self.milvus_db_path)
        self.google_crenditials_path = str(self.google_crenditials_path)
        self.google_token_path = str(self.google_token_path)

file_paths = FilePaths()