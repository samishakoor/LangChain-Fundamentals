from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,     # The maximum number of characters per chunk.
    chunk_overlap=0,    # Number of characters to overlap between chunks. Useful for context continuity.
    separator=''        # separates the text into chunks based on the separator(like new line, paragraph, etc), since we specify '' it will split purely by character count 
)

result = splitter.split_documents(docs)

print(result[1].page_content)