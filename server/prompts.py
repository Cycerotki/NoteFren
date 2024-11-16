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