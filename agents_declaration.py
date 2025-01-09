from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any
import os

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_KEY")


class HealthQuery(TypedDict):
    query_text: str
    category: str
    response: str
    metadata: Dict[str, Any]

class SessionState(TypedDict):
    active: bool
    current_query: str

def setup_health_assistant(model_name: str = "mixtral-8x7b-32768") -> ChatGroq:
    return ChatGroq(model=model_name)

def categorize_health_query(assistant: ChatGroq, query: str) -> str:
    template = """
    As a medical query classifier, determine if this query relates to:
    1. Medication/pharmaceuticals
    2. Medical conditions/symptoms
    
    Query: {user_query}
    
    Respond with only: MEDICATION or CONDITION
    """
    
    classifier = PromptTemplate.from_template(template)
    response = (classifier | assistant).invoke({"user_query": query})
    return "medication" if "medication" in response.content.lower() else "condition"

def process_medication_query(assistant: ChatGroq, query: str) -> str:
    pharmacist_prompt = PromptTemplate.from_template(
        """As a clinical pharmacist, provide detailed information about:
        - Medication class and usage
        - Common indications
        - Important safety considerations
        
        Query: {query}
        """
    )
    return (pharmacist_prompt | assistant).invoke({"query": query})

def process_condition_query(assistant: ChatGroq, query: str) -> str:
    clinician_prompt = PromptTemplate.from_template(
        """As a medical clinician, analyze these symptoms/conditions:
        - Possible conditions
        - Key symptoms to watch
        - When to seek immediate care
        
        Query: {query}
        """
    )
    return (clinician_prompt | assistant).invoke({"query": query})

def create_health_workflow():
    assistant = setup_health_assistant()
    workflow = StateGraph(HealthQuery)
    
    def analyze_query(state: HealthQuery) -> HealthQuery:
        category = categorize_health_query(assistant, state["query_text"])
        return {"query_text": state["query_text"], "category": category}
    
    def handle_medication(state: HealthQuery) -> HealthQuery:
        response = process_medication_query(assistant, state["query_text"])
        return {**state, "response": response}
    
    def handle_condition(state: HealthQuery) -> HealthQuery:
        response = process_condition_query(assistant, state["query_text"])
        return {**state, "response": response}
    
    workflow.add_node("categorize", analyze_query)
    workflow.add_node("medication_handler", handle_medication)
    workflow.add_node("condition_handler", handle_condition)
    
    workflow.add_conditional_edges(
        "categorize",
        lambda x: x["category"],
        {
            "medication": "medication_handler",
            "condition": "condition_handler"
        }
    )
    
    workflow.set_entry_point("categorize")
    workflow.add_edge("medication_handler", END)
    workflow.add_edge("condition_handler", END)
    
    return workflow.compile()

def run_interactive_session():
    workflow = create_health_workflow()
    session_active = True
    
    print("Medical Assistant Interactive Session")
    print("Enter 'exit' to end the session")
    
    while session_active:
        query = input("\nWhat's your health question? > ")
        
        if query.lower() in ('exit', 'quit', 'q'):
            session_active = False
            continue
            
        try:
            result = workflow.invoke({
                "query_text": query,
                "category": "",
                "response": "",
                "metadata": {"timestamp": None}
            })
            
            print("\nAnalysis Results:")
            print("-" * 40)
            print(result["response"])
            
        except Exception as e:
            print(f"\nError processing query: {str(e)}")
            print("Please try rephrasing your question.")


run_interactive_session()