from transformers import pipeline
from langchain import HuggingFaceHub, LLMChain
from langchain_core.prompts import PromptTemplate
import os
from getpass import getpass
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint 
import chainlit as cl


#HUGGINFACEHUB_API_TOKEN=getpass()
#os.environ['HUGGINFACEHUB_API_TOKEN']=HUGGINFACEHUB_API_TOKEN

load_dotenv()

model_id = "gpt2-medium"
conv_model = HuggingFaceEndpoint(
    huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],  
    repo_id=model_id,
    temperature=0.9,
    max_new_tokens=400
)

template= """You are a helpful AI assistant that makes stories  by completing the query provided by the user
{query}
"""

@cl.on_chat_start
def main():
    prompt=PromptTemplate(template=template, input_variables=['query'])

    conv_chain=LLMChain(llm=conv_model,
                        prompt=prompt,
                        verbose=True)
    cl.user_session.set("llm_chain", conv_chain)

@cl.on_message
async def main(message:str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message, callbacks = [cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res["text"]).send()