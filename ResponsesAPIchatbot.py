import os 
from openai import OpenAI
from dotenv import load_dotenv
import json 
from pydantic import BaseModel
from typing import List
load_dotenv()
client=OpenAI(
    api_key=os.getenv("grok_apikey"),
    base_url="https://api.groq.com/openai/v1"
)


history = []
instruction = """You are a research assistant that helps the user investigate topics and keep track of what they learn.

Language:
- Reply in the same language the user writes in.
Research:
- Use web search whenever a question needs current or factual information you may not know for sure (prices, news, recent releases, statistics).
- Don't use web search for simple questions you can answer confidently, or for casual chat.
- Always mention the sources you used.
- If you can't find a reliable answer, say so instead of guessing.

Style:
- Keep answers short and clear. Use bullet points only when listing several items.
- Don't repeat the question back to the user.

Notes:
- You have two tools: save_note and list_notes.
- Call save_note only when the user asks to save or remember something. Don't save notes on your own.
- Call list_notes when the user asks what they saved or wants to review their notes.
- After using a tool, confirm briefly what you did."""
history.append({"role":"developer","content":instruction})

notes = []
def save_notes(title,content):
    notes.append({title:content})
    return "The note has been saved."
def list_notes():
    return notes
tools = [
    {"type":"browser_search" },
    {
        "type":"function",
        "name":"save_notes",
        "description": "Save a note for the user so they can review it later. Use this only when the user explicitly asks to save, remember, or write down something. Do not save information unless the user requests it.",
        "parameters":{
            "type":"object",
            "properties":{
                "title":{"type":"string","description":"Note Title"},
                "content":{"type":"string","description":"Notes details"}
                },
            "required":["title","content"],
            "additionalProperties":False 
        }
            
        
    },
    {
        "type":"function",
        "name":"list_notes",
        "description": "Return all notes the user has saved so far. Use this when the user asks to see, review, or recall their saved notes. Takes no input.",
        "parameters":{
            "type":"object",
            "properties":{},
            "required":[],
            "additionalProperties":False 

        }

    }
]
class format(BaseModel):
    subject : str
    Key_Findings : list[str]
    Sources : list[str]
    opening_Questions : list[str]
    Recommendation : str


while True:
    input_text = input("ask ai")
    
    if input_text.lower() == "exit":
        break
    if input_text.lower() == "/rest":
        history.clear()
        history.append({"role":"developer","content":instruction})
        print("A new conversation has been started.")
        continue
    if input_text.lower() == "/notes":
        input_text = "Show me the saved notes"
    if input_text.lower() == "/brief":
        Specialized_memory=[m for m in history if isinstance(m,dict) and m.get("role") in ["user","assistant"]]
        Specialized_input = Specialized_memory[-10:] + [{"role":"user","content":"Write the research brief based on our conversation."}]
        response_parsed = client.responses.parse(
            model = "openai/gpt-oss-20b",
            instructions = """You are a research assistant. Based on the conversation so far, produce a final research brief.
            
                Rules:
                - Output only a JSON object that matches the required schema. No markdown, no extra text.
                - Use only information that appeared in the conversation. Do not invent facts or sources.
                - subject: the main topic of the research, in one short sentence.
                - Key_Findings: the most important findings, each as one short sentence.
                - Sources: the sources mentioned in the conversation. If none, return an empty list.
                - opening_Questions: questions that remain unanswered or need more research. If none, return an empty list.
                - Recommendation: a short conclusion or suggested next step.
                - Write all values in the same language the user used in the conversation.""",
            input = Specialized_input,
            text_format= format,
            

        )
        response_model = response_parsed.output_parsed
        print(response_model.model_dump_json(indent=2))
       
    else:
        
        history.append({"role":"user","content":input_text})
        response = client.responses.create(
            model = "openai/gpt-oss-20b",
            input = history,
            store = False,
            tools = tools
        )
        print(response.output_text)
        history.append({"role":"assistant","content":response.output_text})
        

        for i in response.output:
            if i.type == "reasoning":
                continue

            if i.type == "function_call" and i.name == "save_notes":
                args = json.loads(i.arguments)
                result = save_notes(**args) 
                history.append(i)
                history.append({"type":"function_call_output","call_id":i.call_id,"output":result})
                response1 = client.responses.create(
                    model = "openai/gpt-oss-20b",
                    input = history,
                    store = False,
                )
                history.append({"role":"assistant","content":response1.output_text})
                print(response1.output_text)

            if i.type == "function_call" and i.name == "list_notes":
                result = list_notes()   #notes[{}]
                history.append(i)
                history.append({"type":"function_call_output","call_id":i.call_id,"output":json.dumps(result)})
                response2 = client.responses.create(
                    model = "openai/gpt-oss-20b",
                    input = history,
                    store = False,
                )
                history.append({"role":"assistant","content":response2.output_text})
                print(response2.output_text)
        

            
        





