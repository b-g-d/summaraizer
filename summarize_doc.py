from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document


llm = OpenAI(temperature=0)

def open_and_split(filepath):
    text_splitter = CharacterTextSplitter()
    with open(filepath) as f:
        text_blob = f.read()
    texts = text_splitter.split_text(text_blob)
    return texts

def main(filepath):
    raw_text = open_and_split(filepath)
    docs = [Document(page_content=t) for t in raw_text]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    print(summary)

if __name__ == "__main__":
    import sys
    import os

    # get and validate command line arguments
    if len(sys.argv) != 2:
        print("Usage: python summarize_doc.py <filepath>")
        sys.exit(1)
    
    filepath = sys.argv[1]

    # validate that file exist
    if not os.path.isfile(filepath):
        print("Error: file does not exist")
        sys.exit(1)

    # verify file is readable and is a text file
    if not os.access(filepath, os.R_OK):
        print("Error: file is not readable")
        sys.exit(1)
    
    if not filepath.endswith(".txt"):
        print("Error: file is not a text file")
        sys.exit(1)

    main(filepath)
    
