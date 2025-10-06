from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Create the HuggingFace endpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)

# Wrap it with ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# Invoke the model
result = model.invoke("What is the capital of Pakistan?")

print(result.content)