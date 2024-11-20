from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


from dotenv import load_dotenv
import os


load_dotenv()


groq_api_key=os.environ['GROQ_API_KEY']


prompt = ChatPromptTemplate.from_template(
        """
You are an expert at creating detailed learning roadmaps in Mermaid syntax.
Based on the user's request, identify the key learning areas.
give me  the complete learning path which must cover the key learning areas.
now generate the  Mermaid-formatted flowchart with the user query as follows  {input}.
 """
    )


llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama-3.1-8b-instant")  #llama3-groq-70b-8192-tool-use-preview
                                                 #llama-3.1-8b-instant
def response(topic):
        llm_chain = LLMChain(
                llm=llm,
                prompt=prompt,
                verbose=True)

        response =llm_chain.invoke({"input":topic})
        return response['text']


if __name__ == "__main__":
        topic="" 
        print(response(topic))
           
                        
            
    
 
 


      
