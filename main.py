from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import SentenceTransformerEmbeddings
embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

llm=CTransformers(model='../blog_generation_llma2/model/llama-2-7b-chat.ggmlv3.q2_K.bin',
                  model_type='llama',
                  config={'max_new_tokens':256,
                          'temperature':.01})



def read_doc(dir):
    file_loader=PyPDFDirectoryLoader(dir)
    docs=file_loader.load() #will have doc pages in list
    return docs

doc=read_doc('../multi_pdf_chatbot/pdf_files')

#print(doc)

def get_chunks(doc):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks=text_splitter.split_documents(doc)
    return chunks

chunks=get_chunks(doc)
#print(chunks)

vectors=embedding_function.embed_query('how are you')
print(f'The length of vector to create index in pinecone:{len(vectors)}')

# pinecone.init(
#     api_key='d6ce1773-07be-44dd-adfc-c863a46b523a',
#     environment='Serverless'
# )
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key='d6ce1773-07be-44dd-adfc-c863a46b523a')


index_name='pdf-chat-bot'

index=Pinecone.from_documents(doc,embedding_function,index_name=index_name)


# template='you are a helpful agent, give the answers for user {inputs} with 10 words '
# prompt=PromptTemplate(
#     input_variables=['inputs'],
#     template=template
# )
# chain=LLMChain(llm=llm,prompt=prompt)
# print(chain.run('what is Machine Learning'))
