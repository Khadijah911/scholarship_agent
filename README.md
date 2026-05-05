 ScholarshipAgent,an AI-Powered Study Abroad Research Assistant

The Story Behind This Project

I have personally gone through the scholarship application process  across different degree levels. And every single time, it felt like I was solving a puzzle with missing pieces.

Finding the right university, figuring out their exact GPA requirement, hunting down application deadlines buried in PDF documents, identifying which professors align with your research interests, and then on top of all that, searching for scholarships that actually apply to you as an international student. It is exhausting. It is time-consuming. And most of the time, you are doing it alone, with dozens of browser tabs open and notes scattered everywhere.

I built this ScholarshipAgent because I wished something like this existed when I was going through it. A tool that does not just point you to a website but actually reads it, extracts what matters, and talks to you like a knowledgeable friend who has done all the research already.

This is for every student who has stayed up late cross-referencing university websites, missed a deadline they did not know existed, or felt overwhelmed by how much information there is and how little of it is organised.

 What the ScholarshipAgent Does:

ScholarshipAgent is a multi-agent AI system built with LangGraph that you interact with through a simple gradio chat interface. Instead of manually searching across dozens of university websites, you just have a conversation and the agents do the work for you.

The system currently has 4 Agents
1.     Supervisor Agent:
 Reads your message and routes it to the right agent automatically 
2.     Opportunity Agent:
| Searches for scholarships matching your field, country, and background
3.     Requirements Agent:
Fetches admission requirements, deadlines, GPA, and documents needed for specific universities |
4.     Faculty Agent:
Finds professors at your target universities whose research matches your interests |

Sample of the Conversations

user: Find me scholarships for computer engineering in Australia as a Nigerian student
Agent: Here are 15 scholarships matched to your profile...

user: What are the application requirements for University of Melbourne?
Agent: University of Melbourne — Masters of Computer Science requires a WAM of 75%, 
       deadline May 31 2026, tuition AUD $62,976/year...

You: Find AI professors at Purdue University
Agent: Here are faculty members in the ECE and CS departments researching AI...

Setup & Installation
1. Clone the repository
bash
git clone https://github.com/Khadijah911/scholarship_agent.git
cd scholarship_agent


 2. Create and activate a virtual environment
bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

 3. Install dependencies
pip install langchain langchain-core langchain-openai langgraph tavily-python gradio beautifulsoup4 requests

4. Set up your API keys

Create a `.env` file in the root folder:

OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here

5. Run the app
python scholarship_agent.py

API Keys You Need
OpenAI 
Tavily 
 Project Structure


Limitations
Tavily free tier has a monthly search limit — the app will pause if exceeded
Some university websites are JavaScript-heavy and may return limited data
Faculty contact details depend on publicly available university pages

 Who This Is For:
International students applying for Masters or PhD programs abroad
Students from Africa, Asia, and other regions navigating foreign university systems
Anyone overwhelmed by the scholarship and admissions research process



This project is not just code. It is built from a real experience of not knowing where to look, missing opportunities, and wishing the process was simpler. If it saves even one student hours of searching — or helps them find a scholarship they would have otherwise missed — then it has 
