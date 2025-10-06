from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

'''
Syntax:

sequence = RunnableSequence(
    step1,   # First runnable (e.g., a PromptTemplate)
    step2,   # Second runnable (e.g., an LLM or ChatModel)
    step3,   # Third runnable (e.g., an OutputParser)
    ...
)

sequence.invoke({'topic':'AI'})

Explanation:

→ RunnableSequence is a sequence of runnables.
→ Executes all runnables sequentially (in order).
→ The output of each step becomes the input to the next step.
→ Returns the final output from the last runnable.
'''

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({'topic':'AI'}))