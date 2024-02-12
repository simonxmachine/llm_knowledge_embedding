
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
valid_password = "abcde111222333"


load_dotenv()

# 1. Vectorize the sales response csv data

loader = CSVLoader(file_path="sample_safeweb_questions.csv")
documents = loader.load()


# Embeddings help turn text into numerical representations of text that can be used 
# for similarity search and to capture meaning and relationships between words and phrases
embeddings = OpenAIEmbeddings()

# FAISS is an open source library  with efficient similarity search in high-dimensional spaces, 
# making it ideal for working with text embeddings.
embedded_knowledge = FAISS.from_documents(documents, embeddings)    

# print (db)


# 2. Do similarity search

def retrieve_info(query):
    similar_response = embedded_knowledge.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]

    return page_contents_array

# customer_mesage = """Is there laundry or air conditioning?"""

# results = retrieve_info(customer_mesage)


# 3. Setup LLMChain & prompts


#Creates a new instance of the ChatOpenAI class to send and receive messages from the OpenAI API.
#Temperature controls the randomness or creativity of the LLM's responses.
#A value of 0 means the model will generate responses that are highly deterministic 
#and consistent with the training data, potentially sacrificing some creativity.
llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo-16k-0613")


#Defines the prompt template that will be used to generate prompts for the LLM.
template = """

You are a machine learning expert that is serving as an agent for a cybersecurity company named
SafeWeb.ai, and our main tool in development is a malicious URL scanner. You are well versed in 
machine learning algorithms and will help introduce our company to website visitors as well as help explain 
in simple terms how our web application and cybersecurity tools were built. I will share a visitor's message with you and you will give the best answer
to the visitor based on past best practices and you will follow ALL of the rules below:

1/ Response should be very similar to the past best practices, 
in terms of length, tone of voice, logical arguments and other details

2/ If the best practice is irrelevant, then try to mimic the style of the best practice to the visitor's message

3/ If the visitor asks a question that is very far from the scope of our company's purpose, kindly divert
the direction of the conversation back to our company. 


Below is a message I received from the visitor:
{message}

Here is a list of best practices of how we normally respond to our visitors in similar scenarios:
{best_practice}

Please write the best friendly and informative response I should send to this visitor:


"""


#Create a new PromptTemplate instance to generate prompts for the LLM.
prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation

@app.route('/')
def say_hello():
    return "Hello, World!"


@app.route('/ask', methods=["POST"])
def generate_response():
    
    password=request.headers.get("password")
    
    if password != valid_password:
        return "Invalid password", 401
    else: 
        data = request.get_json()
        message = data["message"]
        best_practice = retrieve_info(message)
        response = chain.run(message=message, best_practice=best_practice)
        return response


# message = """What school district is this property and are there any tax breaks?"""

# response = generate_response(message)

# print(response)

# 5. Build an app with streamlit


# def main():
    # st.set_page_config(
    #     page_title="Customer response generator", page_icon=":bird:")

    # st.header("Customer response generator :bird:")
    # message = st.text_area("customer message")

    # if message:
    #     st.write("Generating best practice message...")

    #     result = generate_response(message)

    #     st.info(result)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)