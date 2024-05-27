import tiktoken
import pymupdf
from openai import OpenAI
from rich import print
import re


def get_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""

    text = ""
    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def get_token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def shorten_context(context: str, max_tokens: int) -> list:
    """returns a list of strings that are shorter than the max_tokens"""
    
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(context)
    token_count = 0
    shortened_contexts = []
    current_context = ""

    for token in tokens:
        token_count += 1
        if token_count >= max_tokens:
            shortened_contexts.append(current_context)
            current_context = ""
            token_count = 0
        current_context += encoding.decode(token)

    if current_context:
        shortened_contexts.append(current_context)

    return shortened_contexts


def get_surrounding_words(text, target, num_words=5):
    # Find all words in the text
    words = re.findall(r'\w+', text)
    
    # Find the position of the target word
    target_words = re.findall(r'\w+', target)
    target_len = len(target_words)
    
    for i in range(len(words)):
        if words[i:i + target_len] == target_words:
            start = max(i - num_words, 0)
            end = min(i + target_len + num_words, len(words))
            return ' '.join(words[start:end])
    
    return None




def call_openai(prompt, json=True):

    client = OpenAI()
    if json:
        response_format = {"type": "json_object"}
        message_content = f"{prompt} Respond with json"
    else:
        response_format = {"type": "text"}
        message_content = prompt

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format=response_format,
        messages=[
            {"role": "user", "content": message_content},
        ]
    )
    return response

text = get_text_from_pdf("./data/the-fellowship.pdf")

contexts = shorten_context(text, 100000)

for context in contexts:
    prompt = f"""
        Carefully read the included text in it's entirety and answer the question "What is Tony's favorite color?" If you don't find the answer, simply respond with "not found" {context}
        """

    response = call_openai(prompt, json=False)
    print(response.choices[0].message.content)

# Using basic text search and surrounding context
# 
# 
# people = [
#     "Bridgette",
#     "Tony",
#     "Theo",
#     "Ruthie"
# ]
# text = get_text_from_pdf("./data/the-fellowship.pdf")

# for person in people:

#     context = get_surrounding_words(text, person, 200)

#     prompt = f"""
#         Carefully read the included text in it's entirety and answer the question "What is {person} favorite color?" If you don't find the answer, simply respond with "not found" {context}
#         """

#     response = call_openai(prompt, json=False)
#     print(response.choices[0].message.content)

