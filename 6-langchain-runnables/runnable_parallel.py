from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

'''
Syntax:

parallel = RunnableParallel({
    "key1": runnable1,
    "key2": runnable2,
    ...
})

parallel.invoke({'topic':'AI'})

Explanation:

→ RunnableParallel is a dictionary that maps keys to runnables.
→ Executes all runnables (runnable1, runnable2, ...) in parallel.
→ Each key maps to its own runnable (key1 -> Runnable1, key2-> Runnable2, ...).
→ Returns a dictionary where each key has the result of its corresponding runnable.
→ Each runnable executing in parallel will receive the same input that you pass to .invoke(). For example, in the above example, both the runnables will receive the same input {'topic':'AI'}.
'''

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linkedin'])

