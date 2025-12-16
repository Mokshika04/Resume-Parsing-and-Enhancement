import pprint
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel
from typing import List, Optional 
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json

# Loading the PDF
file_path = r"C:\Users\91727\Desktop\Mokshika Pandey_Resume (2).pdf"
loader = PyPDFLoader(file_path, mode="single")
docs = loader.load()
print(len(docs))
pprint.pp(docs[0].page_content)

# Resume Schema
class ExperienceItem(BaseModel):
    company: Optional[str]
    role: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    description: Optional[str]

class EducationItem(BaseModel):
    institution: Optional[str]
    degree: Optional[str]
    start_year: Optional[str]
    end_year: Optional[str]

class ResumeSchema(BaseModel):
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    skills: List[str]
    education: List[EducationItem]
    experience: List[ExperienceItem]

# Parsing
parser = PydanticOutputParser(pydantic_object=ResumeSchema)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                             temperature=0)
parse_prompt = ChatPromptTemplate.from_template("""
You are an expert ATS resume parser. Extract key information from the resume below.

Return only JSON following this schema:
{format_instructions}

Resume text:
```{text}```
""")

messages = parse_prompt.format_messages(
    text=docs[0].page_content,
    format_instructions=parser.get_format_instructions()
)

response = llm.invoke(messages)
parsed_resume = parser.parse(response.content)

print(parsed_resume)

# Resume Enhancement
enhance_prompt = ChatPromptTemplate.from_template("""
You are an expert resume writer.

Rewrite experience bullets to be:
- more action-based
- quantified where possible
- concise (1–2 lines each)
- focused on business impact

Input JSON:
```json
{resume_json}```

Return improved bullets in the SAME JSON format.
""")

messages = enhance_prompt.format_messages(
    resume_json=json.dumps(parsed_resume.model_dump())
)

response = llm.invoke(messages)

# Handle potential empty or invalid JSON response
response_text = response.content.strip()
if not response_text:
    print("Error: Empty response from LLM")
    enhanced_resume = json.loads(json.dumps(parsed_resume.model_dump()))
else:
    try:
        # Try to extract JSON if it's wrapped in markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        enhanced_resume = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing enhanced resume JSON: {e}")
        print(f"Response content: {response.content}")
        enhanced_resume = json.loads(json.dumps(parsed_resume.model_dump()))
print(f"Enhanced Resume:{enhanced_resume}")





