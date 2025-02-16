# backend/response_generator.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict

load_dotenv()

class ResponseGenerator:
    """Handles response generation using LLM and vector retrieval"""
    
    def __init__(self, retriever, history: List[Dict]):
        self.llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_LLM_MODEL"), api_key=os.getenv("GOOGLE_API_KEY"))
        self.retriever = retriever
        self.history = history
        self.prompt = self._create_prompt_template()

    def _create_prompt_template(self):
        """Create the system prompt template"""
        return ChatPromptTemplate.from_messages([
            ("system", """
                You are a professional cooking assistant. Use only the given context to generate responses.  
                Your goal is to provide **clear, human-like** step-by-step cooking guidance.

                **INSTRUCTIONS:**
                1. Respond in natural, conversational language without unnecessary symbols, numbers, or formatting.
                2. When listing recipes or steps, do not include extraneous numbers, labels, or identifiers (e.g., avoid phrases like "Chef 7" or "Chef 3").
                3. Maintain accuracy in the order of steps, ingredients, measurements, and cooking times.
                4. When the user asks for the next step or says 'done/ok/next', provide the next step.
                5. If the user asks for a previous step, provide the previous step.
                6. Once the recipe is completed, say **"Recipe completed."**
                7. Keep track of conversation history to avoid repetition or skipping steps.
                8. If the user asks something unrelated to cooking, politely decline.
                9. Responses should be **clear, natural, and like a human chef explaining.**
                10. Provide **one step at a time** and keep responses concise.
                **Example Interaction:**
                - **User:** "How do I start?"  
                - **Assistant:** "Start by preheating your oven to 180°C and greasing a baking tray."  
                - **User:** "Next?"  
                - **Assistant:** "In a bowl, mix flour, sugar, and butter until crumbly."  

                **Context:** {context}  
                **History:** {history}  
            """),
            ("human", "{question}")
        ])

    def _format_history(self) -> str:
        """Format conversation history for prompt"""
        return "\n".join(
            f"{msg['role'].title()}: {msg['content']}"
            for msg in self.history[-4:]  # Keep last 2 exchanges
        )

    def generate(self, question: str) -> str:
        """Generate response for user query"""
        chain = (
            RunnablePassthrough.assign(
                context=lambda _: self.retriever.get_relevant_documents(question),
                # context=lambda _: self.retriever.invoke({"query": question}),
                history=lambda _: self._format_history()
            )
            | self.prompt
            | self.llm
        )
        return chain.invoke({"question": question}).content
