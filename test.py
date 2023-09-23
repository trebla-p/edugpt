
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st

# load_dotenv()
brwoserless_api_key = st.secrets["BROWSERLESS_API_KEY"]
serper_api_key = st.secrets["SERPER_API_KEY"]

if 'citation_style' not in st.session_state:
    st.session_state['citation_style'] = "APA"

# 1. Tool for search

# sample_course_guide = """
# """

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")

# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to look for recent publications, educational resources, and data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content=f"""You are a professional college educator, who can do detailed research for the purpose of creating an up-to-date course outline for any college course. You will need to provide the following:
            1/ Course Title
            2/ Course Description
            3/ Course Learning Outcomes (CLOs)
            4/ Topics/ Modules and Intended Learning Outcomes (ILOs)
                - each topic requires at least 2 ILOs
                - each topic should be marked by starting with "Topic #", and each ILO should be marked by starting with "ILO #"
            5/ Weekly Activities
                - should be in a table format with the following columns:
                    - Week
                    - Teaching/Learning Activities
                    - Output/Formative Assessment
                    - Assessment Tools (i.e. what will be used to assess the students)
            6/ Grading System
            7/ References in the APA format
            
            Please make sure you complete the objective above with the following rules:
            1/ Before generating your course outline, you should do enough research in scholarly publication sites to gather as much information as possible about the course. You should try looking for results in Google Scholar, ResearchGate, and other scholarly publication sites
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iterations
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ Make sure to keep your sources up-to-date by keeping only sources from the last 5 years. ONLY THE LAST 5 YEARS
            6/ Follow the format above, and keep all the required information. Include activities in your output. Utilize markdown format for your output
            7/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
"""
)
sub_system_message = SystemMessage(
    content=f"""You are a professional college educator, who can do detailed research for the purpose of creating an up-to-date course outline for any college course. 
            For this this task, you will need to look for recent publications, educational resources, and data related to this topic within the given course.
            The purpose of this task is to gather references and resources for the course outline. 
            You will need to provide a list of references and resources for each topic in the APA reference format.
           
            
            Please make sure you complete the objective above with the following rules:
            1/ Before generating your result, you should do enough research in scholarly publication sites to gather as much information as possible about the topic. You should try looking for results in Google Scholar, ResearchGate, and other scholarly publication sites
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 2 iterations
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ Make sure to keep your sources up-to-date by keeping only sources from the last 5 years. ONLY THE LAST 5 YEARS
            6/ Follow the format above, and keep all the required information. Include activities in your output. Utilize markdown format for your output
            7/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

sub_agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": sub_system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-4")

memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=2000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

sub_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=sub_agent_kwargs,
    memory=memory,
) 

def extract_topics(course_outline):
    lines = course_outline.split('\n')  # Split the course outline into lines
    topics = []
    inside_topic = False  # A flag to mark whether we are reading inside a topic section
    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespaces
        if "Topics" in line and "Modules" in line:
            inside_topic = True  # Start reading the topics
            continue
        elif "Weekly Activities:" in line:
            inside_topic = False  # Stop reading when reached the Weekly Activities section
            break
        if inside_topic and line:  # If we are inside a topic section and the line is not empty
            if "Topic" in line:
                topics.append(line)
    return topics

if 'course_outline' not in st.session_state:
    st.session_state['course_outline'] = None

# 4. Use streamlit to create a web app
def main():
    st.set_page_config(page_title="EduGPT", page_icon=":school:")

    st.header("EduGPT Course Outline Generator:school:")
    course_title = st.text_input("Course Title")
    course_description = st.text_area("Course Description")
    col1,col3 = st.columns(2)
    with col1:
        target_audience = st.text_input("Target Student Audience")
    with col3:
        st.session_state['citation_style'] = st.selectbox("Citation Style", ["APA", "MLA", "Chicago", "Harvard", "IEEE"])
    col2,col4 = st.columns(2)
    with col2:
        total_hours = st.number_input("Total Hours", min_value=1, max_value=100, value=54, step=1)
    with col4:
        total_
    if st.button("Generate Course Outline",use_container_width= True):
        query = f"Course Title: {course_title}\nCourse Description: {course_description}\nTarget Student Audience: {target_audience}\nTotal Hours: {total_hours}"
        st.write("Doing research for ", course_title)

        result = agent({"input": query})

        st.subheader("Course Outline Result")
        st.session_state['course_outline'] = result['output']
        st.write(result['output'])

    if st.button("Generate Topic References",disabled=st.session_state['course_outline'] is None, use_container_width=True):
        topics = extract_topics(st.session_state['course_outline'])
        topic_references = []
        for topic in topics:
            st.write("Doing research for ", topic)
            topic_result = sub_agent({"input": topic})
            topic_references.append(topic_result['output'])
        st.subheader("Topic References")
        for topic_reference in topic_references:
            st.write(topic_reference)


if __name__ == '__main__':
    main()
