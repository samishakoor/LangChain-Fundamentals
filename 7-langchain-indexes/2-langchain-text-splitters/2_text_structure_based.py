from langchain.text_splitter import RecursiveCharacterTextSplitter

'''
RecursiveCharacterTextSplitter tries to:

1. Split by double newlines (paragraphs).

2. If still too long → split by single newline (lines).

3. If still too long → split by periods or punctuation (sentences).

4. If still too long → split by spaces (words).

5. If still too long → finally split by characters.

That’s why it’s called recursive — it keeps going down this hierarchy until each piece fits the chunk size limit.
'''

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)