from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("spacex.pdf")

docs = loader.load()

print(len(docs))

print(docs[0].page_content)


#pypdf is best for if pdf contain most of the text otherwise there are much more for image scanned 