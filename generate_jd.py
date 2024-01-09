import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms.replicate import Replicate
from langchain_community.llms.ctransformers import CTransformers
from dotenv import load_dotenv

load_dotenv()

import os

os.environ["REPLICATE_API_TOKEN"]=os.getenv("REPLICATE_API_TOKEN")


def generateJobDescription(job_title, industry, skills):
   
    
    llm = Replicate(model='meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
                        input={ 'temperature': 0.75})
    

    # Define a prompt template for job description
    template = """You are a job description generator specializing in clear and compelling descriptions for tech projects. Craft a job description that captures the following:
                Job Title: {job_title}
                Industry: {industry}
                Key Responsibilities (use bullet points):

                [List specific project-related tasks and responsibilities] Qualifications (use bullet points):
                [List required skills and experience, using phrases like "experience in" or "proficiency in"] Benefits:
                [List any benefits or perks offered by the company] Additional Information:
                [Include any important details, such as work environment, team structure, or opportunities for growth]
                Please use active voice, avoid jargon, and structure the description in a clear and concise manner.
                Your response should only contain the job description,not even a single additional word"""
    
    prompt = PromptTemplate(input_variables=["job_title", "industry", "skills"],
                            template=template)
    input_for_replicate=prompt.format(job_title=job_title, industry=industry,
                                        skills=skills)
    job_description=llm(input_for_replicate)
    

   
    return job_description


st.set_page_config(page_title="Generate Job Description",
                    layout='centered',
                    initial_sidebar_state='collapsed')
st.title('Job Description Generator')

# User inputs for job details
job_title = st.text_input('Enter Job Title')
industry = st.text_input('Enter Industry')
skills = st.text_area('Required Skills')

if st.button('Generate Job Description'):
    generated_description = generateJobDescription(job_title, industry,  skills)
    print(generated_description)
    st.write('Generated Job Description:')
    st.write(generated_description)

