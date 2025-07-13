from enum import Enum
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, Unstructured
from langchain_community.document_loaders import (

)
class Excelsheet():
    pass

class PDF():
    pass 

class Docx():
    pass

class Web():
    pass


class FileType(Enum): 
    excel = 1
    pdf  = 2
    docx = 3
    web = 4 
    
class FilesHandler():

    def __init__(self, file_type):
        try:
            loader = self.files_seletor(FileType[file_type])
        except KeyError as e: 
            raise ValueError(f"file type is not supprt: {e}")


    def files_seletor(self,file_type: FileType): 
        files_type_selctor = {
            FileType.excel: Excelsheet,
            FileType.pdf: PDF, 
            FileType.docx: Docx, 
            FileType.web: Web
        }
        return files_type_selctor[file_type]
    

if __name__ == "__main__":
    h = FilesHandler('docx')

    