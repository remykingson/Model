from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from vector import retriever
model = OllamaLLM(model="mistral")  # Replace with the correct model name
template = """
    You're an expert in answering questions about pizza restaurant
     
    Here are some relevant reviews: {reviews}
     
    Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
while True:
        print("\n\n----------------------------------------")
        question = input("Ask a question (q to quit) : ")
        print("\n\n")
        if question == "q":
            break
        
        reviews = retriever.invoke(question)    
        result = chain.invoke({"reviews": reviews, "question": question})
        print(result)