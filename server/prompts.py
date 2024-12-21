from typing import List, Tuple

def multi_call_prompt(query: str) -> List[Tuple[str, str]]:
    return [("system", "You are a helpful assistant."),
            ("human", query)]

def search_prompt(query: str) -> List[Tuple[str, str]]:
    return [("system", "You are a helpful assistant."),
            ("human", f"Find the answer to this question: {query}")]

def q_n_a_prompt(query: str, info: str) -> List[Tuple[str, str]]:
    return [("system", "You are a helpful assistant."),
            ("human", f"Based on the information given, answer the question concisely. Question: {query}, Information: {info}")]

def summary_prompt(info: str) -> List[Tuple[str, str]]:
    return [("system", "You are a helpful summary assistant."),
            ("human", f"Summarise the information given: {info}")]

def generate_podcast_prompt(topic: str) -> List[Tuple[str, str]]:
    return [("system", "You are a helpful podcast generation assistant."),
            ("human", f" write a podcast that explains about {topic}. List as a transcript with 2 speakers, where speaker 1 is male, speaker 2 is female, in the format 1: (text)\n 2: (text)\n 1:(text)")]