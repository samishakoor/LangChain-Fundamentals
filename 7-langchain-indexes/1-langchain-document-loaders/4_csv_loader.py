from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Social_Network_Ads.csv')  # for each row, a new document is created

docs = loader.load()

print(len(docs))
print(docs[1])