from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


'''
SemanticChunker cuts text whenever the topic or meaning noticeably changes.

Here is how it works:
1. The input text is first split into sentences or small semantic units.
2. Each sentence is converted into a vector embedding using the embedding model you provide.
3. The algorithm then computes the cosine similarity (or semantic distance) between each consecutive pair of sentence embeddings.
4. The semantic chunker finds breakpoints where the semantic similarity drops significantly — i.e., where one topic ends and another begins.
5. The chunker then groups sentences between breakpoints into meaningful sections, where each section focuses on one clear idea or topic.
'''

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)


