import os
from langchain_community.document_loaders import Docx2txtLoader as msword
from langchain.text_splitter import RecursiveCharacterTextSplitter


doc_folder = r"C:\Users\jayba\OneDrive - jaybagchi.com\Personal\Learning"
doc = doc_folder + r"\Paul-Halmos-Measure-Theory-pdf.pdf"
loader = msword(doc)
docs = loader.load()
print(len(docs), "pages")
print("First page chars:", len(docs[0].page_content) if docs else 0)
print(docs[0].page_content[:500] if docs else "NO DOCS")

docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0))

for doc in docs:
    print(doc.page_content)



