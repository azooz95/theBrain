
import re
import json
import pandas as pd 
import numpy as np
from config.config import FilePaths

file_paths = FilePaths()

class TextHandler():
    
    @staticmethod
    def parse_json2dict(data:str, pattern:str = r'```json\s*(\{.*\})\s*```') -> dict: 

        json_match = re.search(r'```json\s*(\{.*\})\s*```', data, re.DOTALL)

        if json_match:
            json_string = json_match.group(1)  # Extract JSON part
            data = json.loads(json_string)  # Convert to dictionary
            return data
        else:
            print("No valid JSON found")
            return None
        
    @staticmethod
    def dict2excel(data: dict, save_name='test'):

        for i in data: 
            if not isinstance(data[i], (list, np.ndarray)):
                data[i] = [data[i]]

        df = pd.DataFrame(data)
        df.to_excel(file_paths.data_base_path + save_name+'.xlsx')

if __name__ == "__main__":
    data = '''```json
                {
                "vendor_name": "PHO CAPITAL",
                "total_amount": "$23.32"
                }
                ```'''
    
    paresing_text = TextHandler()
    data = paresing_text.parse_json2dict(data)
    print(data)
    paresing_text.dict2excel(data)