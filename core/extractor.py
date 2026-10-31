#Actionableitems , decision , questions 

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import httpx
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter 

def _is_rate_limit(error: httpx.HTTPStatusError) -> bool:
    return error.response is not None and error.response.status_code == 429

#temp 0.2 as no extras needed
def get_llm():
    return ChatMistralAI(
        model="mistral-medium-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
        max_retries=int(os.getenv("MISTRAL_MAX_RETRIES", "1")),
        timeout=int(os.getenv("MISTRAL_TIMEOUT", "120")),
    )

def split_transcript(transcript: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)
llm = get_llm()
#each time system prompt is sent to build chain\
#each time system prompt is created for Actionableitems , decision , questions 
def build_chain(system_prompt : str):
    
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def _extract_from_chunks(
    transcript: str,
    map_system_prompt: str,
    combine_system_prompt: str,
    no_result_message: str,
    rate_limit_message: str,
) -> str:
    chunks = split_transcript(transcript)
    if not chunks:
        return no_result_message

    map_chain = build_chain(map_system_prompt)

    try:
        chunk_results = [
            result
            for chunk in chunks
            if (result := map_chain.invoke(chunk).strip())
            and result.casefold() != no_result_message.casefold()
        ]
    except httpx.HTTPStatusError as error:
        if not _is_rate_limit(error):
            raise
        return rate_limit_message

    if not chunk_results:
        return no_result_message

    combined_chain = build_chain(combine_system_prompt)
    try:
        return combined_chain.invoke("\n\n".join(chunk_results))
    except httpx.HTTPStatusError as error:
        if not _is_rate_limit(error):
            raise
        return "\n".join(chunk_results)


def extract_action_items(transcript: str) -> str:
    return _extract_from_chunks(
        transcript,
        map_system_prompt=(
            "You are an expert meeting analyst. Extract every explicit action item "
            "or commitment from this transcript section. For each item include the "
            "task, owner if stated, and deadline if stated; use 'Not specified' "
            "when an owner or deadline is not stated. Do not infer missing details "
            "or include general discussion as an action item. Preserve enough "
            "context to identify the task. Return a numbered list, or exactly "
            "'No action items found.' if this section contains none."
        ),
        combine_system_prompt=(
            "You are an expert meeting analyst. Combine the action items extracted "
            "from every transcript section into one complete list. Remove duplicates "
            "caused by overlapping sections, but keep distinct tasks separate. "
            "Preserve stated owners and deadlines exactly; do not invent or discard "
            "details. Format as a numbered list. If no action items remain, say "
            "'No action items found.'"
        ),
        no_result_message="No action items found.",
        rate_limit_message=(
            "Mistral rate limit reached; action items could not be extracted."
        ),
    )


def extract_key_decisions(transcript: str) -> str:
    return _extract_from_chunks(
        transcript,
        map_system_prompt=(
            "You are an expert meeting analyst. Extract every key decision explicitly "
            "made in this transcript section. Do not treat proposals, opinions, or "
            "unresolved discussion as decisions. Preserve enough context to identify "
            "each decision. Return a numbered list, or exactly 'No key decisions "
            "found.' if this section contains none."
        ),
        combine_system_prompt=(
            "You are an expert meeting analyst. Combine the key decisions extracted "
            "from every transcript section into one complete list. Remove duplicates "
            "caused by overlapping sections, but keep distinct decisions separate. "
            "Do not invent decisions or turn proposals into decisions. Format as a "
            "numbered list. If no decisions remain, say 'No key decisions found.'"
        ),
        no_result_message="No key decisions found.",
        rate_limit_message=(
            "Mistral rate limit reached; key decisions could not be extracted."
        ),
    )


def extract_questions(transcript: str) -> str:
    return _extract_from_chunks(
        transcript,
        map_system_prompt=(
            "You are an expert meeting analyst. Extract every unresolved question "
            "or topic explicitly needing follow-up from this transcript section. "
            "Do not include questions that are answered or resolved in the section, "
            "and do not invent questions. Preserve enough context to identify each "
            "one. Return a numbered list, or exactly 'No open questions found.' "
            "if this section contains none."
        ),
        combine_system_prompt=(
            "You are an expert meeting analyst. Combine the unresolved questions "
            "and follow-up topics extracted from every transcript section into one "
            "complete list. Remove duplicates caused by overlapping sections, but "
            "keep distinct questions separate. Do not invent questions or treat "
            "resolved questions as unresolved. Format as a numbered list. If none "
            "remain, say 'No open questions found.'"
        ),
        no_result_message="No open questions found.",
        rate_limit_message=(
            "Mistral rate limit reached; open questions could not be extracted."
        ),
    )

#to question from the vedio we need RAG to store via vector store