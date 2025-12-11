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





