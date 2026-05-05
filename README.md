# ScholarshipAgent — An AI-Powered Study Abroad Research Assistant

## The Story Behind This Project

I have personally gone through the scholarship application process across different degree levels. And every single time, it felt like I was solving a puzzle with missing pieces.

Finding the right university, figuring out their exact GPA requirement, hunting down application deadlines buried in PDF documents, identifying which professors align with your research interests, and then on top of all that — searching for scholarships that actually apply to you as an international student. It is exhausting. It is time-consuming. And most of the time, you are doing it alone, with dozens of browser tabs open and notes scattered everywhere.

I built this ScholarshipAgent because I wished something like this existed when I was going through it. A tool that does not just point you to a website but actually reads it, extracts what matters, and talks to you like a knowledgeable friend who has done all the research already.

---

## What the ScholarshipAgent Does

ScholarshipAgent is a multi-agent AI system built with LangGraph that you interact with through a simple Gradio chat interface.

### The system currently has 4 agents:

1. **Supervisor Agent**  
   Routes your query to the appropriate agent  

2. **Opportunity Agent**  
   Searches for scholarships based on your profile  

3. **Requirements Agent**  
   Retrieves admission requirements, deadlines, GPA, and documents  

4. **Faculty Agent**  
   Finds professors whose research matches your interests  

---

## Sample Conversations

**User:** Find me scholarships for computer engineering in Australia as a Nigerian student  
**Agent:** Here are 15 scholarships matched to your profile...

**User:** What are the application requirements for University of Melbourne?  
**Agent:** Requires WAM of 75%, deadline May 31 2026, tuition AUD $62,976/year...

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Khadijah911/scholarship_agent.git
cd scholarship_agent