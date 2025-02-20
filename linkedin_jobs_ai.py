import streamlit as st
import warnings

warnings.filterwarnings('ignore')
import os
import tempfile
from utils import get_openai_api_key, get_serper_api_key, get_openai_model_name, get_gemini_api_key, \
    get_gemini_model_name
import google.generativeai as genai
from crewai import Agent, Crew, Task, LLM
from crewai_tools import (
    FileReadTool,
    ScrapeWebsiteTool,
    MDXSearchTool,
    SerperDevTool
)
import mammoth
import markdownify

# Configuration (Environment Variables)
os.environ["GEMINI_API_KEY"] = get_gemini_api_key()
os.environ["GEMINI_MODEL_NAME"] = get_gemini_model_name()
os.environ["SERPER_API_KEY"] = get_serper_api_key()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

my_llm = LLM(
    model=os.environ["GEMINI_MODEL_NAME"],
    api_key=os.environ["GEMINI_API_KEY"]
)

# Initialize Session State

for key in [
    "job_posting_url", "github_url", "personal_writeup", "tailored_resume",
    "interview_materials", "job_application_inputs", "temp_markdown_filename", "temp_resume_filename",
    "temp_interview_filename", "markdown_output", "docx_file"
]:
    if key not in st.session_state:
        st.session_state[key] = "" if key not in ["temp_markdown_filename", "temp_resume_filename", "temp_interview_filename", "job_application_inputs", "docx_file"] else None

#  Helper Functions
def convert_docx_to_markdown(docx_file):
    try:
        html = mammoth.convert_to_html(docx_file).value
        return markdownify.markdownify(html)
    except Exception as e:
        st.error(f"Error converting DOCX: {e}")
        return None


#  CrewAI Agents and Tasks
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Initialize tools AFTER temp file names are in session state (inside the process button logic)
read_resume = None
semantic_search_resume = None

# Initialize tools AFTER the temp file names are in session state
def initialize_tools(file_path): # Pass file path to initialize_tools
    try:
        read_resume = FileReadTool(file_path=file_path)
        semantic_search_resume = MDXSearchTool(mdx=file_path)
        return read_resume, semantic_search_resume
    except Exception as e:
        st.error(f"Error initializing tools: {e}")
        return None, None


# Agent 1: Researcher
researcher = Agent(
    role="Tech Job Researcher",
    goal="Make sure to do amazing analysis on "
         "job posting to help job applicants",
    tools = [scrape_tool, search_tool],
    llm=my_llm,
    verbose=True,
    backstory=(
        "As a Job Researcher, your prowess in "
        "navigating and extracting critical "
        "information from job postings is unmatched."
        "Your skills help pinpoint the necessary "
        "qualifications and skills sought "
        "by employers, forming the foundation for "
        "effective application tailoring."
    )
)

# Agent 2: Profiler
profiler = Agent(
    role="Personal Profiler for Engineers",
    goal="Do incredible research on job applicants "
         "to help them stand out in the job market",
 #   tools = [scrape_tool, search_tool],
 #            read_resume, semantic_search_resume],
    llm=my_llm,
    verbose=True,
    backstory=(
        "Equipped with analytical prowess, you dissect "
        "and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional profiles, laying the "
        "groundwork for personalized resume enhancements."
    )
)

# Agent 3: Resume Strategist
resume_strategist = Agent(
    role="Resume Strategist for Engineers",
    goal="Find all the best ways to make a "
         "resume stand out in the job market.",
 #   tools = [scrape_tool, search_tool],
          #   read_resume, semantic_search_resume],
    llm=my_llm,
    verbose=True,
    backstory=(
        "With a strategic mind and an eye for detail, you "
        "excel at refining resumes to highlight the most "
        "relevant skills and experiences, ensuring they "
        "resonate perfectly with the job's requirements."
    )
)

# Agent 4: Interview Preparer
interview_preparer = Agent(
    role="Engineering Interview Preparer",
    goal="Create interview questions and talking points "
         "based on the resume and job requirements",
   # tools = [scrape_tool, search_tool],
           #  read_resume, semantic_search_resume],
    llm=my_llm,
    verbose=True,
    backstory=(
        "Your role is crucial in anticipating the dynamics of "
        "interviews. With your ability to formulate key questions "
        "and talking points, you prepare candidates for success, "
        "ensuring they can confidently address all aspects of the "
        "job they are applying for."
    )
)

# Task for Researcher Agent: Extract Job Requirements
research_task = Task(
    description=(
        "Analyze the job posting URL provided ({job_posting_url}) "
        "to extract key skills, experiences, and qualifications "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of job requirements, including necessary "
        "skills, qualifications, and experiences."
    ),
    agent=researcher,
    async_execution=True
)

# Task for Profiler Agent: Compile Comprehensive Profile
profile_task = Task(
    description=(
        "Compile a detailed personal and professional profile "
        "using the GitHub ({github_url}) URLs, and personal write-up "
        "({personal_writeup}). Utilize tools to extract and "
        "synthesize information from these sources."
    ),
    expected_output=(
        "A comprehensive profile document that includes skills, "
        "project experiences, contributions, interests, and "
        "communication style."
    ),
    agent=profiler,
    async_execution=True
)

# The task will not run until it has the output(s) from those tasks.


# Task for Resume Strategist Agent: Align Resume with Job Requirements
resume_strategy_task = Task(
    description=(
        "Using the profile and job requirements obtained from "
        "previous tasks, tailor the resume to highlight the most "
        "relevant areas. Format the resume using Markdown. "
        "Employ tools to adjust and enhance the "
        "resume content. Make sure this is the best resume even but "
        "don't make up any information. Update every section, "
        "including the initial summary, work experience, skills, "
        "and education. All to better reflect the candidates "
        "abilities and how it matches the job posting."
    ),
    expected_output=(
        "An updated resume that effectively highlights the candidate's "
        "qualifications and experiences relevant to the job."
    ),
 #   output_file=st.session_state.get("temp_resume_filename", "tailored_resume.md"),  # Correct path
    context=[research_task, profile_task],
    agent=resume_strategist
)

# Task for Interview Preparer Agent: Develop Interview Materials
interview_preparation_task = Task(
    description=(
        "Create a set of potential interview questions and talking "
        "points based on the tailored resume and job requirements. "
        "Format the questions and talking points using Markdown. "
        "Utilize tools to generate relevant questions and discussion "
        "points. Make sure to use these question and talking points to "
        "help the candidate highlight the main points of the resume "
        "and how it matches the job posting."
    ),
    expected_output=(
        "A document containing key questions and talking points "
        "that the candidate should prepare for the initial interview."
    ),
  #  output_file=st.session_state.get("temp_interview_filename", "interview_materials.md"), # Correct path
    context=[research_task, profile_task, resume_strategy_task],
    agent=interview_preparer
)

#Crew
job_application_crew = Crew(
    agents=[researcher, profiler, resume_strategist, interview_preparer],
    tasks=[research_task, profile_task, resume_strategy_task, interview_preparation_task],
    verbose=True
)

# Streamlit App
st.title("Job Application Assistant")

# Input fields (bind to session state)
#job_posting_url = st.text_input("Job Posting URL:", value=st.session_state.job_posting_url)
#github_url = st.text_input("GitHub URL (Optional):", value=st.session_state.github_url)
#personal_writeup = st.text_area("Personal Write-up or Cover Letter:", height=200, value=st.session_state.personal_writeup)

# Initialize session state FIRST
# Initialize job_application_inputs in session state FIRST
for key in [
    "job_posting_url", "github_url", "personal_writeup", "tailored_resume",
    "interview_materials", "temp_markdown_filename", "temp_resume_filename",
    "temp_interview_filename", "job_application_inputs", "markdown_output", "docx_file"
]:
    if key not in st.session_state:  # Crucial check to prevent re-initialization
        st.session_state[key] = "" if key not in ["temp_markdown_filename", "temp_resume_filename", "temp_interview_filename", "job_application_inputs", "docx_file"] else None



job_posting_url = st.text_input("Job Posting URL:", key="job_posting_url")
github_url = st.text_input("GitHub URL (Optional):", key="github_url")
personal_writeup = st.text_area("Personal Write-up or Cover Letter:", height=200, key="personal_writeup")


# DOCX to Markdown Conversion
uploaded_file = st.file_uploader("Upload DOCX Resume (Optional)", type="docx")

if uploaded_file:
    st.session_state.docx_file = uploaded_file

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as temp_file:  # Create temp file
        temp_markdown_filename = temp_file.name  # Get temp file name
        markdown_output = convert_docx_to_markdown(uploaded_file)
        if markdown_output:
            temp_file.write(markdown_output)  # Write to the temp file
            st.session_state.temp_markdown_filename = temp_markdown_filename  # Store the temp file name
            st.session_state.markdown_output = markdown_output  # Store content

            # No download button
            #st.download_button(
            #          label="Download Converted Markdown",
            #           data=st.session_state.markdown_output,
            #          file_name="converted_resume.md",
            #           mime="text/markdown"
            #       )
            #st.markdown(st.session_state.markdown_output)  # Display in Streamlit


# Process button
if st.button("Generate Job Application Materials"):
    if job_posting_url and personal_writeup:
        st.session_state.job_application_inputs = {
            'job_posting_url': job_posting_url,
            'github_url': github_url,
            'personal_writeup': personal_writeup
        }

        with st.spinner("Generating..."):
            try:
                # Create temporary files (only if they don't exist)
                if st.session_state.temp_resume_filename is None:
                    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as f:
                        st.session_state.temp_resume_filename = f.name
                if st.session_state.temp_interview_filename is None:
                    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as f:
                        st.session_state.temp_interview_filename = f.name

                if st.session_state.temp_markdown_filename:  # Check if temp_markdown_filename exists (after DOCX conversion)
                    file_path = st.session_state.temp_markdown_filename
                elif st.session_state.temp_resume_filename:  # Fallback to temp_resume_filename if no DOCX conversion
                    file_path = st.session_state.temp_resume_filename
                else:
                    file_path = None  # No file to process

                if file_path:  # Process only if file exists

                    # Initialize tools AFTER temp file names are available
                    read_resume, semantic_search_resume = initialize_tools(file_path)

                    if read_resume and semantic_search_resume: # Check if tools were initialized
                        profiler.tools = [scrape_tool, search_tool, read_resume, semantic_search_resume]
                        resume_strategist.tools = [scrape_tool, search_tool, read_resume, semantic_search_resume]
                        interview_preparer.tools = [scrape_tool, search_tool, read_resume, semantic_search_resume]
                    else:
                        st.warning(
                            "Resume file not uploaded or could not be processed. Proceeding without resume analysis.")

                    job_application_crew.tasks[2].output_file = st.session_state.temp_resume_filename
                    job_application_crew.tasks[3].output_file = st.session_state.temp_interview_filename

                    result = job_application_crew.kickoff(inputs=st.session_state.job_application_inputs)

                    # Read from temporary files and store in session state
                    with open(st.session_state.temp_resume_filename, "r") as f:
                        st.session_state.tailored_resume = f.read()
                    with open(st.session_state.temp_interview_filename, "r") as f:
                        st.session_state.interview_materials = f.read()

                    #  Display content (using st.session_state)
                    st.subheader("Tailored Resume")
                    if st.session_state.tailored_resume:
                        st.markdown(
                            f"""
                            <div style="word-wrap: break-word; overflow-y: auto; max-height: 600px;">  
                                {st.session_state.tailored_resume}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write("No tailored resume generated yet.")

                    st.subheader("Interview Materials")
                    if st.session_state.interview_materials:  # Check if content exists
                        st.markdown(
                            f"""
                            <div style="word-wrap: break-word; overflow-y: auto; max-height: 600px;">  
                                {st.session_state.interview_materials}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write("No interview materials generated yet.")
                  #  st.experimental_rerun()
                else:
                    st.warning("No resume file provided. Please upload a DOCX or provide a markdown file.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

    else:
        st.warning("Please fill in all mandatory input fields.")



# Clear button
if st.button("Clear"):
    for key in [
        "job_posting_url", "github_url", "personal_writeup", "tailored_resume",
        "interview_materials", "temp_markdown_filename", "temp_resume_filename", "temp_interview_filename",
        "job_application_inputs", "markdown_output", "docx_file"  # Clear all relevant keys
    ]:

        st.session_state[key] = "" if key not in ["temp_markdown_filename", "temp_resume_filename", "temp_interview_filename",
                                                  "job_application_inputs", "docx_file"] else None

        # Clear uploaded file if present
    if "docx_file" in st.session_state and st.session_state.docx_file is not None:
        st.session_state.docx_file = None  # Clear the uploaded file object itself.

    st.empty()  # This will clear any rendered widgets like text inputs
    st.session_state.clear()