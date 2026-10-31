from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
import httpx

import os

def get_llm():
    return ChatMistralAI(
        model="mistral-medium-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
        max_retries=int(os.getenv("MISTRAL_MAX_RETRIES", "1")),
        timeout=int(os.getenv("MISTRAL_TIMEOUT", "120")),
    )
#chunking
# split the transcript into chunks of 3000 characters with an overlap of 200 characters
def split_transcript(transcript: str)-> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )
    return splitter.split_text(transcript)

def _is_rate_limit(error: httpx.HTTPStatusError) -> bool:
    return error.response is not None and error.response.status_code == 429

def _local_summary(transcript: str) -> str:
    sentences = [sentence.strip() for sentence in transcript.split(".") if sentence.strip()]
    if not sentences:
        return "No summary could be generated."
    return "\n".join(f"- {sentence}." for sentence in sentences[:8])

def summarize(transcript:str)-> str:
    llm= get_llm()
  # creating a prompt template for summarizing a portion of a meeting transcript concisely
    map_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ]
    )
# creating a chain that takes the transcript, applies the prompt, 
# and generates a summary using the LLM
    map_chain = map_prompt | llm | StrOutputParser()
#store the transcripts chunks
    chunks = split_transcript(transcript)
#store the summeries from chain in chunk_summaries by sending each chunk 
# recursively as context window may exceed
    try:
        chunk_summaries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]
    except httpx.HTTPStatusError as error:
        if not _is_rate_limit(error):
            raise
        return _local_summary(transcript)
#adding summeries from every chunks
    combined = "\n\n".join(chunk_summaries)
#generating a combined summary using AI
    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )
#finally sending the combined summary to the chain
    try:
        return combined_chain.invoke(combined)
    except httpx.HTTPStatusError as error:
        if not _is_rate_limit(error):
            raise
        return "\n".join(f"- {summary}" for summary in chunk_summaries)

def generate_title(transcript:str)->str:
    llm=get_llm()

    title_chain = (
         RunnablePassthrough() | RunnableLambda(lambda x:{"text":x})|
         ChatPromptTemplate.from_messages([(
             "system",
             "Based on the meeting transcript,generate a  short professional meeting title"
             "(max 8 words). Only  return the tiltle,nothing else."
         ),
         ("human", "{text}")
         ])
         |llm
        |StrOutputParser()

    )
    try:
        return title_chain.invoke(transcript[:2000])
    except httpx.HTTPStatusError as error:
        if not _is_rate_limit(error):
            raise
        first_sentence = transcript.strip().split(".", 1)[0].strip()
        return (first_sentence[:80] or "Meeting Transcript").rstrip(" .,!?")