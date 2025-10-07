from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()  # lazy_load fetches one file at a time and returns a generator of Document objects

for document in docs:
    print(document.metadata)