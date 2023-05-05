from io import StringIO
import streamlit as st
import requests 
import json
import time
import random
import time
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
import getpass
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import torch
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


st.set_page_config(page_title="CATALYST", 
                   page_icon="https://endlessicons.com/wp-content/uploads/2012/12/fountain-pen-icon-614x460.png",
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )

def p_title(title):
    st.markdown(f'<h2 style="text-align: left; color:#F63366; font-size:28px;">{title}</h2>', unsafe_allow_html=True)

#########
#SIDEBAR
########

st.sidebar.header('CATALYST - I would like to :crystal_ball:')
nav = st.sidebar.radio('',['☀️Go to homepage', '🤖Explain Code', '🦾Search Docs', '👨🏾‍💻Code Similarity'])
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
os.environ['OPENAI_API_KEY'] = "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"

@st.cache_resource
def load_explainer_vectorstore():
    #with open("Files\explainer.pkl", "rb") as f:
    explainer_db = CPU_Unpickler(open("Files\explainer.pkl","rb")).load()
    retriever = explainer_db.as_retriever()
    return retriever

@st.cache_resource
def load_doc_vectorstore():
    #with open("vectorstore.pkl", "rb") as f:
        #python_db = CPU_Unpickler.load(f)
    with open('vectorstore.pkl', 'rb') as file:
        unpickler = pickle.Unpickler(file)
        python_db = unpickler.load()
    retriever = python_db.as_retriever()
    return retriever

@st.cache_resource
def faiss_loader1():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf = HuggingFaceEmbeddings(model_name=model_name)
    new_db = FAISS.load_local("faiss_index1", hf)
    return new_db.as_retriever(search_type = "similarity", search_kwargs = {"k" : 10})

@st.cache_resource
def faiss_loader2():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf = HuggingFaceEmbeddings(model_name=model_name)
    new_db = FAISS.load_local("faiss_index2", hf)
    return new_db.as_retriever(search_type = "similarity", search_kwargs = {"k" : 10})

@st.cache_resource
def get_qa_model(_retriever):
    qa = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature = 0), 
                                                        chain_type="stuff", 
                                                        retriever=_retriever,
                                                        reduce_k_below_max_tokens=True,)
    return qa

@st.cache_resource
def get_results(question, _qa):
    result = qa({"question": question})
    return result




if nav == '☀️Go to homepage':
    st.title("📒 Welcome to Catalyst 🤖 ")
    #st.markdown("<h1 style='text-align: center; color: white; font-size:28px;'>Welcome to Catalyst!</h1>", unsafe_allow_html=True)
    #st.markdown("<h3 style='text-align: center; font-size:56px;'<p>&#129302;</p></h3>", unsafe_allow_html=True)
    #st.markdown("<img src="https://endlessicons.com/wp-content/uploads/2012/12/fountain-pen-icon-614x460.png" style="zoom: 50%;" />", unsafe_allow_html=True)
    #col1, col2, col3 = st.columns([6, 2, 6])
    #col2.image("https://d1nhio0ox7pgb.cloudfront.net/_img/g_collection_png/standard/512x512/fountain_pen.png", width = 150)
    #st.markdown("<h3 style='text-align: center; color: grey; font-size:20px;'>Dont know what an object is in Python or how list comprehensions work or even how fibonacci numbers are calculated in Python? Use our tool to write code and make it explain to you and learn better! !</h3>", unsafe_allow_html=True)

    #st.markdown('___')
    #st.write(':point_left: Use the menu at left to select a task (click on > if closed).')
    #st.markdown('___')
    #st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>What is this App about?<b></h3>", unsafe_allow_html=True)
    #st.write("")
    #st.write("Python Code Explainer")     
    # Libraries


    # Title
    st.title('CATALYST')


    st.write(
        """
Are you a Python coder who wants to level up your skills and productivity? Do you wish you had a smart and friendly assistant who could help you understand, debug and optimize your code? If yes, then you need to check out **CATALYST**, the ultimate web app for Python enthusiasts!

CATALYST is a revolutionary web app that harnesses the power of Large Language Models (LLMs) to provide you with natural language explanations about your Python code. Whether you are writing a simple script or a complex project, CATALYST can help you learn from your code, spot errors, improve readability and performance, and discover new ways to solve problems.

But that's not all! CATALYST also lets you search a documentation corpus with querying a stored vector database with an LLM. This means you can find relevant and up-to-date information about any Python topic or function in seconds. No more wasting time browsing through endless pages of documentation or tutorials. Just type in your query and get instant results.

And if you are looking for some coding challenges to test your skills and learn from others, CATALYST has you covered too. CATALYST can get similar questions if a Python question is posted from Leetcode using Codebert Embeddings and Langchain. This way, you can compare different approaches, learn best practices, and get feedback from other coders.

CATALYST is more than just a web app. It's a holistic learning experience that will make you a better Python coder in no time. So what are you waiting for? Sign up today and get ready to code like a pro!
        """
    )

    st.subheader('Methodology')
    st.write(
        """
        We have developed a web app that can summarize (Natural language Explanations), Search Docs over a Vector Database and Find similar questions by pasting a solution and getting results who have solutions which are similar to the one you posted. 
        
        Our Code Explainer  uses low-rank adaptation (LoRA) to fine-tune Large Language Models (LLMs) like LLaMA on consumer hardware. We call it CATALYST and it is based on PEFT, a library that lets you take various transformers-based language models and fine-tune them using LoRA. Our web app can produce outputs comparable to the Stanford Alpaca model, which is an instruct model of similar quality to text-davinci-003.
        To fine-tune our model we followed these steps:
        We first download the LLaMA open source model weights and convert them to a transformers-compatible format using our script fine-tune-alpaca.py
        Then, we prepare our instruction data in a JSON file with the following format: {"instruction": "Explain this Code to me: ", "code": "def add(a,b): return a + b", "Explanation" : "This is a python code to add two numbers a and b"}.
        Finally, you need to run our fine-tuning script finetune_alpaca.py with the following arguments: python finetune.py --model_name_or_path weights/llama-7b --train_file alpaca_data.json --output_dir output --do_train --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --num_train_epochs 3 --save_steps 1000 --save_total_limit 1
        This will fine-tune the model on the instruction data using LoRA and save the best checkpoint in the output directory. 
        

        Our document search engine is built using **FAISS**, **Langchain** and **Scrapy** to index and retrieve documents from docs.python.org. We call it **Python Doc Search** and it is based on **Large Language Models (LLMs)** to embed and query the documents in a vector space. Our search engine can find relevant and up-to-date information about any Python topic or function in seconds.
        To build our search engine, we followed these steps:
        First, we used **Scrapy**, a web scraping framework, to crawl and scrape documents from docs.python.org. We saved the documents in an Output folder of text files.
        Next, we used **Langchain**, a natural language processing library, to split the documents into chunks of sentences or paragraphs. We also used Langchain to embed the documents, i.e., convert them from natural language to text embeddings using LLMs. We saved the embeddings in a numpy file with the following format: `{"url": "https://docs.python.org/3/library/math.html", "title": [0.1, 0.2, ..., 0.9], "content": [[0.1, 0.2, ..., 0.9], [0.2, 0.3, ..., 1.0], ...]}`. 
        Finally, we used **FAISS**, a library for efficient similarity search and clustering of dense vectors, to store and index the embeddings in a FAISS index. We also used FAISS to retrieve the most similar documents given a query vector. We embedded the query vector using the same LLMs as the documents. We used our script `search.py` to perform the search and display the results. For example: `python search.py --query "How do threads work in Python?"`
        
        Our Last feature i.e Find Similar Code uses a similar approach as of the search engine but the embedding model is different. We used Codebert embeddings which are very good at contextually capturing code semantics.
        We use a simple function to link hyperlinks with code and we use a query code example to search through the vector database and return the related hyperlinks
        """
    )

    st.subheader('Future Works')
    st.write(
        """
        There is a lot to do in this application. We want to increase the verbosity of explanations and make them more accurate whilst staying at the same or even better latency. 
        One could achieve this using Specialized Inference Servers such as the NVIDIA Triton Inference Server  and a cluster of GPUs for online inference.
        One could also add more documentation pickle files, our system was becoming memory intensive so we just created a vector store for python docs and code embeddings.
        But with our custom script you can create a DocVectorstore for any documentations, take pandas for example. 
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info('**Ashwin: ashwinr@vt.edu**', icon="💻")
    with c2:
        st.info('**Swanand: swanandsv@vt.edu**', icon="💻")
    with c3:
        st.info('**Darshan: dvekaria@vt.edu**', icon="💻")


if nav == '🤖Explain Code':    
    st.markdown("<h4 style='text-align: center; color:grey;'>Accelerate knowledge with Catalyst &#129302;</h4>", unsafe_allow_html=True)
    st.text('')
    st.title("📒 Code Explainer 🤖 ")
    st.text('')

    source = st.radio("How would you like to start? Choose an option below",
                          ("I want to input some text", "I want to upload a file"))
    st.text('')
    
    s_example = """
                class Solution(object):
                    def isValid(self, s):
                        stack = []
                        mapping = {")": "(", "}": "{", "]": "["}
                        for char in s:
                            if char in mapping:
                            top_element = stack.pop() if stack else '#'
                            if mapping[char] != top_element:
                                return False
                            else:
                                stack.append(char)
                        return not stack
                """
    if source == 'I want to input some text':
        input_su = st.text_area("Use the example below or input your own code with appropriate indentations)", value=s_example, max_chars=10000, height=330)
        if st.button('Explain Code'):
            with st.spinner('Processing...'):
                #time.sleep(2)
                st.markdown('___')
                st.write('Results Produced by Alpaca-Lora')
                    #convert string to json  
                #response = requests.post("https://ashwinr-pythoncodeexplainer.hf.space/run/predict", json={
	                #"data": [str(input_su),]}).json()
                response = requests.post("https://tloen-alpaca-lora.hf.space/run/predict", json={
                        "data": [
                            "Explain this code line by line to me:",
                            input_su,
                            1,
                            0.75,
                            40,
                            4,
                            300,
                        ]
                    }).json()
                data = response["data"]
                st.success("".join(data))
                
                st.balloons()

    if source == 'I want to upload a file':
        file = st.file_uploader('Upload your file here',type=['txt'])
        if file is not None:
            with st.spinner('Converting your code to explanations...'):
                    time.sleep(2)
                    stringio = StringIO(file.getvalue().decode("utf-8"))
                    string_data = stringio.read()
                    time.sleep(2)
                    st.markdown('___')
                    st.write('Results Produces by CodeT5')
                    #convert string to json 
                    jsonfile = {
	                "data": [
                        string_data
	                        ]
                    }
                    #response = requests.post("https://ashwinr-pythoncodeexplainer.hf.space/run/predict", json=jsonfile).json()
                    response = requests.post("https://tloen-alpaca-lora.hf.space/run/predict", json={
                        "data": [
                            "Explain this code to me:",
                            string_data,
                            0.1,
                            0.75,
                            40,
                            4,
                            300,
                        ]
                    }).json()
                    data = response["data"]
                    st.caption("")
                    st.success(response)
                    st.balloons()

if nav == '🦾Search Docs':
    st.markdown("<h4 style='text-align: center; color:grey;'>Search for Specific Documentation &#129302;</h4>", unsafe_allow_html=True)
    st.text('')
    st.title("📒 Search Documentations 🤖 ")
    st.text('')
    input_su = st.text_area("Hi! I am here to help you with your doc search. Query anything you want to know about Python.", max_chars=1000, height=330)
    if st.button('Search Docs'):
            with st.spinner('Searching For Relevant Docs...'):
                #time.sleep(2)
                st.markdown('___')
                #st.write('Results Produced!')
                    #convert string to json  
                #doc_vector_store = load_doc_vectorstore()
                doc_vector_store = faiss_loader1()
                qa = get_qa_model(doc_vector_store)
                res = get_results(input_su, qa)
                st.caption("Hurray!")
                st.success(res)
                st.balloons()

if nav == '👨🏾‍💻Code Similarity':
    st.markdown("<h4 style='text-align: center; color:grey;'>Search for Specific Documentation &#129302;</h4>", unsafe_allow_html=True)
    st.text('')
    st.title("📒 DSASearch Engine 🤖 ")
    st.text('')
    text_input = st.text_area("Enter a Code Example", value = 
                str("""
                class Solution:
                    def subsets(self, nums: List[int]) -> List[List[int]]:
                        outputs = []
                        def backtrack(k, index, subSet):
                            if index == k:
                                outputs.append(subSet[:])
                                return
                            for i in range(index, len(nums)):
                                backtrack(k, i + 1, subSet + [nums[i]])
                        for j in range(len(nums) + 1):
                            backtrack(j, 0, [])
                        return outputs
                """).strip(), height = 330
                )
    if st.button('Find Similar Questions'):
            with st.spinner('Processing...'):
                #time.sleep(2)
                @st.cache_resource
                def get_db():
                    with open("Files\codesearchdb.pickle", "rb") as f:
                        db = CPU_Unpickler(f).load()
                        print(db)
                    return db


                def get_similar_links(query, db, embeddings):
                    embedding_vector = embeddings.embed_query(query)
                    docs_and_scores = db.similarity_search_by_vector(embedding_vector, k = 10)
                    hrefs = []
                    for docs in docs_and_scores:
                        html_doc = docs.page_content
                        soup = BeautifulSoup(html_doc, 'html.parser')
                        href = [a['href'] for a in soup.find_all('a', href=True)]
                        hrefs.append(href)
                    links = []
                    for href_list in hrefs:
                        for link in href_list:
                            links.append(link)
                    return links
                
                @st.cache_resource
                def get_hugging_face_model():
                    model_name = "mchochlov/codebert-base-cd-ft"
                    hf = HuggingFaceEmbeddings(model_name=model_name)
                    return hf

                embedding_vector = get_hugging_face_model()
                db = get_db()
                query = text_input
                answer = get_similar_links(query, db, embedding_vector)
                for link in set(answer):
                    st.write(link)

                st.markdown('___')
                st.caption("Hurray!")
                st.balloons()
    

