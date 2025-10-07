from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

'''
1. WebBaseLoader uses BeautifulSoup under the hood to parse the HTML and extract the text.
2. It is useful when the website contains mostly text-based or static content like blogs, articles, etc.
3. It does not handle JavaScript-heavy websites well (use SeleniumURLLoader for that).
'''

model = ChatOpenAI(model='gpt-4o-mini')

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.samishakoor.me/'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'What is this content about?', 'text':docs[0].page_content}))