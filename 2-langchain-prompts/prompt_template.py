from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# detailed way
template = PromptTemplate(
    template='Greet this person in 5 languages. The name of the person is {name}',
    input_variables=['name']
)

# fill the values of the placeholders
prompt = template.invoke({'name':'sami'}) # method 1
print(prompt) # text='Greet this person in 5 languages. The name of the person is sami'

# prompt = template.format(name='sami') # method 2
# print(prompt) # Greet this person in 5 languages. The name of the person is sami


result = model.invoke(prompt)

print(result.content)

