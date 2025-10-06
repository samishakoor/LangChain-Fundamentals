from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

'''
Syntax:

branch = RunnableBranch(
    (conditional_function1, runnable1),            # if 
    (conditional_function2, runnable2),            # else if
    ...,
    default_runnable                               # else
)

branch.invoke({'topic':'Russia vs Ukraine'})

Explanation:

→ conditional_function1: A function that takes the input (whatever is passed to invoke) and returns True or False.
→ runnable1: The runnable to be executed if the conditional_function1 returns True.
→ default_runnable: The default runnable similar to else in if-else, If none of the conditional_functions return True, then the default_runnable will be executed.

→ In RunnableBranch, every conditional function and every runnable associated with it receives the same input that you pass to .invoke(). For example, in the above example, both the conditional function and the runnable will receive the same input {'topic':'Russia vs Ukraine'}.
'''

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser), 
    RunnablePassthrough()   # this is the default runnable similar to else in if-else
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))



