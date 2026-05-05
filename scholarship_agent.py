

import os
import json
import time
import re
import requests
from datetime import datetime
from typing import TypedDict, List, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv() 

from bs4 import BeautifulSoup
from tavily import TavilyClient

from langchain.schema import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import gradio as gr
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = openai_api_key
llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=tavily_api_key)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# The states

class ScholarshipSystemState(TypedDict):
    user_profile: Dict[str, Any]
    profile_complete: bool
    last_question_field: Optional[str]
    messages: List[Dict[str, str]]
    current_query: str
    user_intent: str
    scholarships_found: List[Dict[str, Any]]
    selected_scholarships: List[str]
    universities: List[str]
    university_requirements: Dict[str, Dict]
    matched_faculty: Dict[str, List[Dict]]
    search_results: List[Dict[str, str]]
    fetched_pages: List[Dict[str, str]]
    is_followup_response: bool
    summary_sent: bool
    current_agent: str
    current_step: str
    is_complete: bool
    application_progress: Dict[str, str]
    errors: List[str]
    tracked_applications: List[Dict[str, Any]]
    tracker_export_path: Optional[str]

# profile questions
PROFILE_QUESTIONS = {
    "country":            "Which country are you from?",
    "field_of_study":     "What field are you studying?",
    "degree_level":       "Are you applying for a Masters or PhD?",
    "research_interests": "What are your research interests? (e.g. AI, NLP)"
}

# tools


def search_scholarships(field_of_study, country, degree_level, target_country="USA", num_results=5):
    queries = [
    f"{field_of_study} {degree_level} scholarships {target_country} for {country} students",
    f"fully funded {field_of_study} {degree_level} programs {target_country}",
    f"{target_country} universities fully funded {degree_level} {field_of_study} international students",
    f"best {target_country} universities {field_of_study} {degree_level} full funding stipend"
]
    print(f" Target country: {target_country}")

    all_scholarships = []

    for i, query in enumerate(queries, 1):
        print(f"\n Query {i}/4: {query}")

        response = tavily_client.search(
            query=query,
            max_results=num_results,
            search_depth="advanced"
        )

        results = response.get("results", [])

        if results:
            for result in results:
                scholarship_data = {
                    "name":           result["title"],
                    "url":            result["url"],
                    "description":    result.get("content", "No description available"),
                    "source":         result.get("url", "Unknown"),
                    "field":          field_of_study,
                    "degree_level":   degree_level,
                    "target_country": target_country,
                    "raw_content":    result.get("content", "")
                }
                all_scholarships.append(scholarship_data)
                print(f" {result['title'][:70]}")
        else:
            print(f" No results for this query")
    return all_scholarships

results = search_scholarships(
    field_of_study="Computer Science",
    country="Nigeria",
    degree_level="Masters",
    target_country="USA",
    num_results=3
)

if results:
    print(f"\n Sample result:")
    print(f"  Name:        {results[0]['name']}")
    print(f"  URL:         {results[0]['url']}")
    print(f"  Description: {results[0]['description'][:150]}")

# PROMPTS

SCHOLARSHIP_ADVISOR_PROMPT = """You are an intelligent Scholarship Advisor Agent.
Your role is to help users discover, filter, rank, and recommend scholarships based on their needs.

Your responsibilities:
1. UNDERSTAND USER INTENT - Extract constraints such as number, funding type, country, field, requirements
2. FILTER RESULTS - Apply all user constraints strictly
3. RANK & RECOMMEND - Rank by funding level, relevance, deadline urgency, benefits
4. PERSONALIZE - Tailor recommendations to user background
5. ALWAYS LIST THE ACTUAL UNIVERSITY OR ORGANIZATION NAME - Never use the webpage title as the scholarship name

CRITICAL FORMATTING RULES:
- Always extract and display the ACTUAL university or institution name (e.g. "MIT", "University of Toronto")
- Never use webpage titles like "Top 25 Scholarships in Canada" as a scholarship name
- If the source is a list page, extract individual scholarships/universities FROM that page
- For each result show: University/Organization name, country, funding type, key benefits, why it matches
- If a detail is not in the data say "not specified — check official website"
- Never invent requirements or benefits

EXAMPLE OF GOOD OUTPUT:
1. University of Toronto — Fully Funded MSc in Computer Science
   Country: Canada
   Funding: Fully funded (tuition + stipend)
   Benefits: Research assistantship, living allowance
   Why it matches: ...

EXAMPLE OF BAD OUTPUT:
1. Top 25 Fully Funded Scholarships In Canada 2026
   (This is a webpage title, not a real scholarship name — never do this)
"""
REQUIREMENTS_ADVISOR_PROMPT = """You are an intelligent University Requirements Advisor
helping a student understand and compare admission requirements clearly.

Your job is NOT to dump raw data - you must:
1. PRESENT each university's requirements in clean readable format
2. HIGHLIGHT urgent or strict items - tight deadlines, high GPA cuts, many documents
3. COMPARE universities side by side if more than one is requested
4. FLAG missing data honestly - never guess or fill gaps
5. END with a 1-2 sentence action tip for the student

STRICT DATA RULES:
- Only state details explicitly present in the data provided
- If a field is null or missing say "not specified — verify at official website"
- Never invent deadlines, GPA scores, or document requirements
- Never use your training knowledge to fill gaps
- Add " verify at official website" next to any GPA figure since sources vary
- Treat every university as if you have never heard of it
- Do NOT paraphrase or reinterpret data - use the exact values from the data
"""

# The nodes
def supervisor_node(state):
    messages = state["messages"]
    last_user_message = messages[-1]["content"] if messages else ""
    history_messages = [
        m for m in messages[-6:]
        if m["role"] in ("user", "assistant")
    ]
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}"
        for m in history_messages
    )

    prompt = f"""
    A student is using a scholarship advisor chatbot.
    Classify their message into exactly one of these intents:
    "find_scholarships"  -> they want scholarship recommendations OR are asking
                             a follow-up question about scholarships already shown
                             (e.g., "find me scholarships in Germany",
                             "which of those don't require IELTS?",
                             "tell me more about the second one")
    "check_requirements" -> they want university admission requirements
                             (e.g., "what do I need to apply to MIT")
    "find_faculty"       -> they want to find professors or faculty to contact
                             (e.g., "find faculty at Harvard working on NLP")
    "general_chat"      ->  greeting, thanks, or unrelated question
                             (e.g., "hi", "thanks", "how are you?")
    "track_application" -> they want to add, update, show or export their application tracker
                             (e.g., "add scholarship 1 to my tracker",
                             "show my tracker", "update scholarship 2 to Submitted",
                             "export my tracker")

    Use the conversation history to understand context, a short follow-up message
    like "which ones are fully funded?" should be classified based on what was
    discussed before, not just the message alone.

    Conversation history:
    {history_text}

    Current message: "{last_user_message}"

    Reply with ONLY the intent string, nothing else.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        intent   = response.content.strip().strip('"')
        valid_intents = ["find_scholarships", "check_requirements", "find_faculty", "general_chat","track_application"]
        if intent not in valid_intents:
            print(f" Supervisor returned invalid intent '{intent}' defaulting to general_chat")
            intent = "general_chat"

    except Exception as e:
        print(f" Supervisor LLM failed: {e}  defaulting to general_chat")
        intent = "general_chat"

    print(f" Supervisor detected intent: {intent}")

    state["messages"].append({
        "role":    "system",
        "content": f"Supervisor node triggered  detected intent: {intent}"
    })
    state["user_intent"] = intent

    return state

def route_from_supervisor(state):
    intent = state.get("user_intent")
    if intent == "find_scholarships":
        return "opportunity_agent"
    if intent == "check_requirements":
        return "requirements_agent"
    if intent == "find_faculty":
        return "faculty_agent"
    if intent == "track_application":
        return "tracker_agent"
    return END


# OPPORTUNITY AGENT
def opportunity_agent(state: ScholarshipSystemState) -> ScholarshipSystemState:

    user_profile   = state["user_profile"]
    field_of_study = user_profile.get("field_of_study", "Computer Science")
    country        = user_profile.get("country", "Nigeria")

    current_message = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "")
    history_messages = [
        m for m in state["messages"][-6:]
        if m["role"] in ("user", "assistant")
    ]
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:1500] if m['role'] == 'assistant' else m['content'][:300]}"
        for m in history_messages
    )

    followup_prompt = f"""
    Is the user asking a follow-up question about scholarships already shown to them,
    or are they asking for a new scholarship search?
    Reply with ONLY one of: "followup" or "new_search"

    Conversation history:
    {history_text}
    Current message: "{current_message}"
    """
    try:
        followup_response = llm.invoke([HumanMessage(content=followup_prompt)])
        is_followup = followup_response.content.strip().lower() == "followup"
    except Exception:
        is_followup = False

    if is_followup:
        print(" Follow-up detected, advising from existing results")
        existing = state.get("scholarships_found", [])

        if not existing:
            print(" Follow-up detected but no existing results, treating as new search")
            is_followup = False
        else:
            messages_for_llm = [
                SystemMessage(content=SCHOLARSHIP_ADVISOR_PROMPT),
                HumanMessage(content=f"""
The user's profile:
- Country: {country}
- Field of Study: {field_of_study}
- Degree Level: {user_profile.get("degree_level", "Masters")}
- Research Interests: {user_profile.get("research_interests", "")}

Conversation history:
{history_text}
Current question: "{current_message}"
Raw scholarship data:
{json.dumps(existing[:8], indent=2)}

Instructions:
- Answer ONLY what the user asked
- Use the conversation history to understand context
- If a detail like IELTS requirement or deadline is not in the data, say "not specified — check the official website"
- Be concise and specific
""")
            ]
            try:
                answer = llm.invoke(messages_for_llm)
                state["messages"].append({
                    "role":    "assistant",
                    "content": answer.content.strip()
                })
                state["is_followup_response"] = True
                print(" Follow-up answered by advisor")
                return state
            except Exception as e:
                print(f" Follow-up answer failed: {e} - falling through to new search")
                is_followup = False

    if not is_followup:
        print(" New search  extracting target country...")

        prompt = f"""
        A user is asking about scholarships. Extract the target country they want to study in.
        If no country can be determined, reply with exactly: NONE
        Reply with ONLY the country name or NONE, nothing else.
        Conversation history:
        {history_text}

        Current message: "{current_message}"
        """
        try:
            response       = llm.invoke([HumanMessage(content=prompt)])
            target_country = response.content.strip()
            if not target_country or target_country.upper() == "NONE":
                target_country = user_profile.get("country", "USA")
                print(f" Target country: '{target_country}' (profile fallback)")
            else:
                print(f" Target country: '{target_country}' (from message)")
        except Exception as e:
            print(f" Could not extract target country: {e}")
            target_country = user_profile.get("country", "USA")

        state["current_agent"] = "opportunity"
        state["current_step"]  = "searching"

        raw_scholarships = search_scholarships(
            field_of_study=field_of_study,
            country=country,
            degree_level=user_profile.get("degree_level", "Masters"),
            target_country=target_country
        )

        state["scholarships_found"] = raw_scholarships
        print(f"Raw results: {len(raw_scholarships)} scholarships fetched")

        messages_for_llm = [
            SystemMessage(content=SCHOLARSHIP_ADVISOR_PROMPT),
            HumanMessage(content=f"""
The user's profile:
- Country: {country}
- Field of Study: {field_of_study}
- Degree Level: {user_profile.get("degree_level", "Masters")}
- Research Interests: {user_profile.get("research_interests", "")}

The user asked: "{current_message}"
Raw scholarship results:
{json.dumps(raw_scholarships, indent=2)}

Instructions:
Analyze and filter based on the user's request.
Apply constraints strictly.
Rank and justify your recommendations.
For each scholarship include: name, country, funding type, key benefits, why it matches.
If a detail is not in the data say "not specified — check the official website".
Do not hallucinate requirements or benefits.
""")]
        try:
            advisor_response = llm.invoke(messages_for_llm)
            state["advisor_response"] = advisor_response.content.strip()
            state["messages"].append({
                "role":    "assistant",
                "content": advisor_response.content.strip()
            })
            print(" Advisor LLM generated recommendation")
        except Exception as e:
            print(f" Advisor LLM failed: {e}")
            state["advisor_response"] = ""

        state["current_step"] = "complete"
        print(" OpportunityAgent done")

    return state
# REQUIREMENTS SEARCH

def search_requirements(university=None, program=None, degree_level="Masters", state=None, num_results=5):

    if state is None:
        state = {}
    if "messages" not in state:
        state["messages"] = []
    if "requirements_query" not in state:
        state["requirements_query"] = {}

    info         = state["requirements_query"]
    universities = info.get("universities", [])
    programs     = info.get("programs", [])
    degree_level = info.get("degree_level") or degree_level

    if university:
        new_unis = [u.strip() for u in re.split(r',|and', university) if u.strip()]
        universities.extend(new_unis)
    if program:
        programs.append(program.strip())

    info["universities"]        = list(set(universities))
    info["programs"]            = list(set(programs))
    info["degree_level"]        = degree_level
    state["requirements_query"] = info

    if not universities:
        state["messages"].append({
            "role":    "assistant",
            "content": "Please tell me which universities you'd like to check requirements for."
        })
        return state

    results = []

    for uni in universities:
        for prog in programs or ["Computer Science"]:
            print(f"\n Searching requirements for: {uni} — {prog} ({degree_level})")

             
            queries = [
                f"{uni} {prog} {degree_level} admission requirements GPA documents",
                f"{uni} {prog} {degree_level} application deadline 2025 2026",
                f"{uni} graduate school {prog} tuition funding financial aid",
                f"{uni} {prog} masters apply online application portal URL",
            ]

            raw_results = []
            seen_urls   = set()

            for query in queries:
                try:
                    response = tavily_client.search(
                        query=query,
                        max_results=num_results,
                        search_depth="advanced",
                        include_answer=True        
                    )

                    tavily_answer = response.get("answer", "")
                    if tavily_answer and len(tavily_answer) > 20:
                        raw_results.append({
                            "title":   f"[Tavily Answer] {query[:60]}",
                            "url":     "tavily_synthesised",
                            "snippet": tavily_answer
                        })
                        print(f"  Tavily answer: {tavily_answer[:120]}")

                    for r in response.get("results", []):
                        url     = r.get("url", "")
                        snippet = r.get("content", "")
                        if len(snippet) < 20 or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        raw_results.append({
                            "title":   r.get("title", ""),
                            "url":     url,
                            "snippet": snippet
                        })
                        print(f"   {r.get('title', '')[:70]}")
                except Exception as e:
                    print(f"   Search failed for '{query[:50]}': {e}")

            # ── Fetch full page content from top URLs ─────────────────────────
            full_page_texts = []
            seen_fetch_urls = set()

            for r in raw_results[:6]:
                url = r.get("url", "")

                if not url or url == "tavily_synthesised" or url in seen_fetch_urls:
                    continue
                if url.endswith(".pdf") or "/pdf/" in url.lower():
                    print(f"   ⏭️  Skipping PDF: {url[:60]}")
                    continue
                seen_fetch_urls.add(url)
                try:
                    page_resp = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
                    page_resp.raise_for_status()
                    soup = BeautifulSoup(page_resp.text, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "header"]):
                        tag.decompose()
                    page_text = soup.get_text(separator="\n", strip=True)[:15000]

     
                    if len(page_text) > 300:
                        full_page_texts.append({"url": url, "text": page_text})
                        print(f"   Fetched ({len(page_text)} chars): {url[:60]}")
                    else:
                        print(f"   Thin page skipped ({len(page_text)} chars): {url[:60]}")

                except Exception as e:
                    print(f"  Could not fetch {url[:60]}: {e}")

          
            for p in full_page_texts:
                print(f"\n--- PAGE: {p['url'][:60]} ---")
                print(p['text'][:500])
                print("--- END ---")

          
            combined_sources = []

            if full_page_texts:
                combined_sources.extend([
                    {"source": "full_page", "url": p["url"], "text": p["text"]}
                    for p in full_page_texts
                ])

          
            for r in raw_results:
                combined_sources.append({
                    "source":  "snippet",
                    "url":     r["url"],
                    "title":   r["title"],
                    "text":    r["snippet"]
                })

          
            requirements = {}
            if combined_sources:
                extraction_prompt = f"""
Extract ONLY information explicitly stated in the source data below.
If a field is not clearly stated in ANY of the sources, use null.
Do NOT use any prior knowledge about this university.
Do NOT guess or infer — only extract what is directly written.

Pay special attention to:
- Deadlines: look for month/day/year patterns like "December 1", "December 15, 2025"
- GPA: look for numbers like "3.5", "3.0/4.0", "85/100"
- Apply URL: look for links containing "apply", "application", "grad.apply" etc.
- Funding: look for mentions of TA, RA, fellowship, stipend, financial aid
- Tuition: look for dollar amounts per credit, per semester, or per year

University: {uni}
Program: {prog} ({degree_level})

Source data (full pages and snippets combined):
{json.dumps(combined_sources, indent=2)}

Return ONLY this exact JSON object — no markdown fences, no commentary:
{{
  "university": "{uni}",
  "program": "{prog}",
  "degree_level": "{degree_level}",
  "deadline": null,
  "gpa_requirement": null,
  "test_scores": null,
  "required_documents": [],
  "tuition": null,
  "funding": null,
  "apply_url": null,
  "notes": null
}}
"""
                try:
                    response = llm.invoke([
                        SystemMessage(content=(
                            "You are a precise data extraction assistant. "
                            "Return only valid JSON. No markdown, no preamble, no commentary."
                        )),
                        HumanMessage(content=extraction_prompt)
                    ])
                    raw = response.content.strip()

                    if raw.startswith("```"):
                        parts = raw.split("```")
                        raw   = parts[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    raw = raw.strip()

                    requirements = json.loads(raw)
                    print(f" Structured data parsed for {uni}")

                except Exception as e:
                    print(f" Extraction failed for {uni}: {e}")
                    requirements = {
                        "university":         uni,
                        "program":            prog,
                        "degree_level":       degree_level,
                        "deadline":           None,
                        "gpa_requirement":    None,
                        "test_scores":        None,
                        "required_documents": [],
                        "tuition":            None,
                        "funding":            None,
                        "apply_url":          None,
                        "notes":              "Structured parse failed — check official website",
                        "raw_snippet":        raw_results[0]["snippet"][:300] if raw_results else ""
                    }
            else:
                requirements = {
                    "university":         uni,
                    "program":            prog,
                    "degree_level":       degree_level,
                    "deadline":           None,
                    "gpa_requirement":    None,
                    "test_scores":        None,
                    "required_documents": [],
                    "tuition":            None,
                    "funding":            None,
                    "apply_url":          None,
                    "notes":              "No results found from search"
                }

            results.append(requirements)

    state["messages"].append({
        "role":    "system",
        "content": f"Search requirements agent triggered for: {', '.join(universities)}"
    })

    state["messages"].append({
        "role":    "assistant",
        "content": json.dumps(results)
    })

    state["requirements_query"]["universities"] = []
    state["requirements_query"]["programs"]     = []

    return state
# REQUIREMENTS AGENT
def requirements_agent(state: ScholarshipSystemState) -> ScholarshipSystemState:
    import json
    import re
    print(" RequirementsAgent starting...")

    user_profile    = state["user_profile"]
    program         = user_profile.get("field_of_study", "Computer Science")
    current_message = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        ""
    )

    # Recent conversation history for context
    history_messages = [
        m for m in state["messages"][-6:]
        if m["role"] in ("user", "assistant")
    ]
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}"
        for m in history_messages
    )

    
    existing    = state.get("university_requirements", {})
    is_followup = False

    if existing:
        followup_prompt = f"""
Is the user asking a follow-up question about university requirements already shown,
or are they asking about requirements for a new university?
Reply with ONLY one of: "followup" or "new_search"

Conversation history:
{history_text}

Current message: "{current_message}"
"""
        try:
            followup_response = llm.invoke([HumanMessage(content=followup_prompt)])
            result     = followup_response.content.strip().lower().strip('"').strip("'")
            is_followup = result == "followup"
            print(f" Follow-up classification: '{result}' → is_followup={is_followup}")
        except Exception as e:
            print(f" Follow-up check failed: {e}")
            is_followup = False

    if is_followup:
        print(" Follow-up detected -answering from existing data, no new search")
        followup_answer_prompt = f"""
A user is asking a follow-up question about university requirements already retrieved.
Use ONLY the requirements data below to answer — do not use prior knowledge.
Be specific and conversational. If asked about deadlines, funding, or documents,
pull the exact details from the data.
If a detail is not in the data say "not specified — check the official website".

Conversation history:
{history_text}

Current message: "{current_message}"

Requirements data:
{json.dumps(existing, indent=2)}
"""
        try:
            answer = llm.invoke([
                SystemMessage(content=REQUIREMENTS_ADVISOR_PROMPT),
                HumanMessage(content=followup_answer_prompt)
            ])
            state["messages"].append({
                "role":    "assistant",
                "content": answer.content.strip()
            })
            print(" Follow-up answered")
        except Exception as e:
            print(f"Follow-up answer failed: {e}")
            state["messages"].append({
                "role":    "assistant",
                "content": "Sorry, I couldn't retrieve the follow-up answer. Please try rephrasing."
            })
        return state   

    #  Extract universities from the message
    extract_prompt = f"""
A student is asking about university requirements.
Extract all university names mentioned in their message or implied by conversation history.
Return ONLY a JSON array of university names, nothing else.
If no universities are mentioned or implied, return an empty array [].

Conversation history:
{history_text}

Current message: "{current_message}"
"""
    try:
        response = llm.invoke([HumanMessage(content=extract_prompt)])
        raw      = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        extracted_universities = json.loads(raw.strip())
        extracted_universities = [u.strip() for u in extracted_universities]
        print(f" Universities extracted: {extracted_universities}")
    except Exception as e:
        print(f" Could not extract universities: {e}")
        extracted_universities = []

    universities = extracted_universities or state.get("universities", [])
    state["universities"] = []   

    if not universities:
        print(" No universities found in message")
        state["messages"].append({
            "role":    "assistant",
            "content": (
                "Please tell me which universities you'd like to check requirements for. "
                "For example: 'What are the requirements for Stanford and MIT?'"
            )
        })
        return state

    state["current_agent"] = "requirements"
    state["current_step"]  = "searching"

    # Fetch requirements for each university
    all_requirements = {}

    for university in universities:
        print(f"\n Getting requirements for: {university}")

        sub_state = search_requirements(
            university=university,
            program=program,
            degree_level=user_profile.get("degree_level", "Masters"),
            state={"messages": [], "requirements_query": {}}
        )

        assistant_messages = [
            m for m in sub_state["messages"] if m["role"] == "assistant"
        ]

        if assistant_messages:
            try:
                content = assistant_messages[-1]["content"].strip()

                import re
                json_match = re.search(r'\[\s*\{', content)
                if not json_match:
                    raise ValueError(f"No JSON array found in response for {university}")

                parsed = json.loads(content[json_match.start():])
                all_requirements[university] = parsed[0]
                print(f"    Parsed structured data for {university}")

            except Exception as e:
                print(f"    Could not parse requirements for {university}: {e}")
                all_requirements[university] = {
                    "university":      university,
                    "program":         program,
                    "degree_level":    user_profile.get("degree_level", "Masters"),
                    "deadline":        None,
                    "gpa_requirement": None,
                    "required_documents": [],
                    "tuition":         None,
                    "funding":         None,
                    "apply_url":       None,
                    "notes":           "Could not parse results — check official website"
                }
        else:
            all_requirements[university] = {
                "university": university,
                "notes":      "No results returned from search"
            }

    
    existing_requirements = state.get("university_requirements", {})
    existing_requirements.update(all_requirements)
    state["university_requirements"] = existing_requirements
    state["current_step"]            = "complete"

    print(f"\n RequirementsAgent done — fetched {len(all_requirements)} universities")

    #  Format final advisor response 
    requirements_data = state["university_requirements"]

    advisor_user_prompt = f"""
The student's profile:
Field: {program}
Degree level: {user_profile.get("degree_level", "Masters")}
Country: {user_profile.get("country", "Not specified")}

The student asked: "{current_message}"

Requirements data:
{json.dumps(requirements_data, indent=2)}

Instructions:
- Present each university separately using this exact format:

 University Name
 Deadline: ....
 GPA: ...  verify at official website
 Documents: .....
 Tuition: ...
 Funding: ...
 Apply: ...
 Notes: .....

- If a field is null or missing write: "not specified, verify at official website"
- If multiple universities requested, add a brief comparison table at the end
- End with a 1-sentence action tip for the student
"""

    try:
        advisor_response = llm.invoke([
            SystemMessage(content=REQUIREMENTS_ADVISOR_PROMPT),
            HumanMessage(content=advisor_user_prompt)
        ])
        state["messages"].append({
            "role":    "assistant",
            "content": advisor_response.content.strip()
        })
        print(" Advisor response generated")

    except Exception as e:
        print(f" Advisor LLM failed: {e} - using fallback formatter")

        # Fallback: format without LLM
        fallback_lines = []
        for uni, req in requirements_data.items():
            deadline  = req.get("deadline")           or "not specified - verify at official website"
            gpa       = req.get("gpa_requirement")    or "not specified - verify at official website"
            docs      = req.get("required_documents") or []
            tuition   = req.get("tuition")            or "not specified - verify at official website"
            funding   = req.get("funding")            or "not specified - verify at official website"
            apply_url = req.get("apply_url")          or "check official website"
            notes     = req.get("notes")              or ""

            if isinstance(deadline, dict):
                deadline = " | ".join(f"{k.title()}: {v}" for k, v in deadline.items())

            fallback_lines.append(
                f" **{uni}**\n"
                f" Deadline: {deadline}\n"
                f" GPA: {gpa}  verify at official website\n"
                f" Documents: {', '.join(docs) if docs else 'not specified — verify at official website'}\n"
                f" Tuition: {tuition}\n"
                f" Funding: {funding}\n"
                f" Apply: {apply_url}\n"
                f" Notes: {notes}"
            )

        state["messages"].append({
            "role":    "assistant",
            "content": "\n\n".join(fallback_lines)
        })

    return state

# Faculty search
def search_faculty_by_research_area(
    university: str,
    program: str,
    user_research_area: str,
    num_results: int = 5,
) -> list[dict]:
    queries = [
        f"{university} {program} professor {user_research_area}",
        f"{university} faculty {user_research_area} research lab",
        f"{university} {program} {user_research_area} research group members",
        f"{university} {user_research_area} faculty site:ox.ac.uk OR site:cam.ac.uk OR site:ed.ac.uk",
        f"{university} professor {user_research_area} profile email",
    ]

    candidates = []
    seen_urls  = set()

    for query in queries:
        print(f"  [Step 0] {query}")

        response = tavily_client.search(
            query=query,
            max_results=num_results,
            search_depth="advanced"
        )

        organic = response.get("results", [])
        # chiosing to prioritise university domain results first
        organic.sort(key=lambda r: (
            0 if f"cs.{university.lower()}.edu" in r.get("url", "") else 1
        ))

        for r in organic:
            url  = r.get("url", "")
            name = r.get("title", "")

            profile_keywords = ["/faculty/", "/people/", "/profile/", "/person/",
                                 "/staff/", "/~", "/bio/", "/researcher/", "/academic/"]
            is_profile = any(kw in url.lower() for kw in profile_keywords)
            is_scholar = "scholar.google.com/citations" in url
            is_uk_academic = ".ac.uk" in url

            if (is_profile or is_scholar or is_uk_academic) and url not in seen_urls:
                seen_urls.add(url)
                candidates.append({"name": name, "url": url, "source": "direct_search"})
                print(f" {name[:70]}")

    print(f" Step 0 found {len(candidates)} direct candidates")
    return candidates



def resolve_scholar_url(name, university, program):
    clean_name = name.split(" - ")[0].split(" | ")[0].strip()
    query      = f"{clean_name} {university} {program} faculty profile"
    print(f"  Re-searching for department profile: {clean_name}")

    profile_keywords = ["/faculty/", "/people/", "/profile/", "/person/",
                        "/staff/", "/~", "/bio/"]
    try:
        response = tavily_client.search(
            query=query,
            max_results=5,
            search_depth="advanced"
        )
        for r in response.get("results", []):
            url = r.get("url", "")
            if ".edu" not in url:
                continue
            if any(kw in url.lower() for kw in profile_keywords):
                if "scholar.google" not in url:
                    print(f"   Found: {url[:80]}")
                    return url
    except Exception as e:
        print(f"  Re-search failed: {e}")

    return None

def search_faculty_directory(university, program):
    queries = [
        f"{university} {program} faculty directory",
        f"{university} {program} people faculty",
        f"{university} {program} faculty by name",
        f"{university} {program} all faculty members", ]

    all_results = []
    seen_urls   = set()

    for query in queries:
        print(f"   [Step 1] {query}")
        response = tavily_client.search(
            query=query,
            max_results=5,
            search_depth="advanced" )
        for r in response.get("results", [])[:3]:
            url = r.get("url", "")
            if url and url not in seen_urls:
                if "wikipedia.org" in url or "linkedin.com" in url:
                    continue
                seen_urls.add(url)
                all_results.append({
                    "title":   r.get("title", ""),
                    "url":     url,
                    "snippet": r.get("content", ""),
                })
                print(f"  {r['title'][:70]}")

    return all_results

def fetch_page_text(url, timeout=10):
    blocked_domains = [
        "researchgate.net", "linkedin.com", "getprog.ai",
        "academia.edu", "semanticscholar.org",
    ]
    if any(domain in url for domain in blocked_domains):
        print(f"  Skipping blocked domain: {url[:60]}")
        return ""
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:8000]
    except Exception as e:
        print(f" Could not fetch {url}: {e}")
        return ""

def extract_faculty_links(url):
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  Could not load directory page: {e}")
        return []

    profile_keywords = [
        "/faculty/", "/people/", "/profile/", "/person/",
        "/staff/", "/member/", "/bio/", "/about/",
    ]
    base_domain = "/".join(url.split("/")[:3])
    candidates = []
    seen_urls  = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if not text or href.startswith("mailto:") or href == "#":
            continue
        if href.startswith("/"):
            href = base_domain + href
        if any(kw in href.lower() for kw in profile_keywords) and href not in seen_urls:
            seen_urls.add(href)
            candidates.append({"name": text, "url": href, "source": "directory"})

    print(f"    Extracted {len(candidates)} total profile links from directory")
    return candidates

def llm_shortlist_batched(candidates, user_research_area, top_n_per_batch=3, final_top_n=8, batch_size=30):
    if len(candidates) <= final_top_n:
        print(f" Only {len(candidates)} candidates — skipping shortlist")
        return candidates

    shortlist = []
    n_batches = (len(candidates) + batch_size - 1) // batch_size
    print(f" Processing {len(candidates)} candidates in {n_batches} batch(es) of {batch_size}")

    for batch_idx in range(n_batches):
        batch = candidates[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        names = "\n".join([f"{i+1}. {c['name']}" for i, c in enumerate(batch)])

        prompt = f"""
You are helping find university faculty who work SPECIFICALLY on "{user_research_area}".
Pick ONLY the ones whose PRIMARY research area is directly related to "{user_research_area}".
Reply ONLY with a comma-separated list of numbers, e.g.: 2,5,7
If nobody qualifies, reply with exactly: NONE

Faculty list:
{names}
"""
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw      = response.content.strip()
            if raw and raw.upper() != "NONE":
                indices = [int(x.strip()) - 1 for x in raw.split(",") if x.strip().isdigit()]
                picked  = [batch[i] for i in indices if 0 <= i < len(batch)]
                shortlist.extend(picked)
                print(f" Batch {batch_idx+1}/{n_batches}: picked {[p['name'] for p in picked]}")
            else:
                print(f" Batch {batch_idx+1}/{n_batches}: no relevant candidates found")
        except Exception as e:
            print(f" Batch {batch_idx+1} shortlisting failed: {e}")
            shortlist.extend(batch[:top_n_per_batch])

    seen   = set()
    unique = []
    for item in shortlist:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)

    print(f"Batched shortlist: {len(unique)} unique candidates across all batches")
    return unique[:final_top_n]

def llm_extract_faculty_profile(page_text, profile_url):
    if not page_text.strip():
        return None

    prompt = f"""
Extract the following fields from this faculty profile page text.
Return ONLY a JSON object with these exact keys:
  - name               (string)
  - title              (string)
  - email              (string, or null)
  - research_interests (list of strings, specific to this person, not generic department tags)

If a field is not found, use null or empty list.
Do NOT include any explanation, just the JSON.

Page text:
{page_text}
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw      = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        data["profile_url"] = profile_url
        return data
    except Exception as e:
        print(f" Extraction failed for {profile_url}: {e}")
        return None

def llm_rank_faculty(faculty_list, user_research_area, top_n=5):
    if not faculty_list:
        return []

    faculty_summary = json.dumps(
        [{"index": i, "name": f.get("name", "Unknown"), "title": f.get("title", ""),
          "research_interests": f.get("research_interests", [])}
         for i, f in enumerate(faculty_list)],
        indent=2,)

    prompt = f"""
The user's research interest is: "{user_research_area}"

Select the top {top_n} most relevant faculty and return a JSON array.
Each item must have:
  - index         (the original index number)
  - match_score   (integer 1-10)
  - match_reason  (1-2 sentence explanation)

Return ONLY the JSON array, no other text.

Faculty:
{faculty_summary}
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw      = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        rankings = json.loads(raw.strip())
        ranked = []
        for r in rankings:
            idx = r.get("index")
            if idx is not None and 0 <= idx < len(faculty_list):
                entry = faculty_list[idx].copy()
                entry["match_score"]  = r.get("match_score", 0)
                entry["match_reason"] = r.get("match_reason", "")
                ranked.append(entry)
        ranked.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        return ranked[:top_n]
    except Exception as e:
        print(f"  Ranking failed: {e}")
        return faculty_list[:top_n]

def llm_extract_flat_directory(page_text, directory_url):
    if not page_text.strip():
        return []

    prompt = f"""
This is a university faculty directory page listing multiple faculty members on a single page.
Extract ALL faculty members and return a JSON array.

Each item must have:
  - name               (string)
  - title              (string)
  - email              (string or null)
  - research_interests (list of strings)
  - profile_url        (use "{directory_url}" as the value)

CRITICAL: Only include faculty whose names appear explicitly in the page text. Do NOT hallucinate names.
Return ONLY the JSON array, no explanation.

Page text:
{page_text}
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw      = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        faculty_list = json.loads(raw.strip())
        if not isinstance(faculty_list, list):
            return []
        print(f" Flat extraction found {len(faculty_list)} faculty on the page")
        return faculty_list
    except Exception as e:
        print(f"  Flat directory extraction failed: {e}")
        return []

def _fetch_direct_candidates(candidates, university, program):
    extracted = []
    for candidate in candidates[:6]:
        url  = candidate["url"]
        name = candidate.get("name", "")
        if "scholar.google.com" in url:
            resolved = resolve_scholar_url(name, university, program)
            if resolved:
                url = resolved
            else:
                continue
        page_text = fetch_page_text(url)
        if len(page_text) < 200:
            continue
        profile_data = llm_extract_faculty_profile(page_text, url)
        if profile_data and profile_data.get("name"):
            profile_data["source"] = "direct_search"
            extracted.append(profile_data)
            print(f" {profile_data['name']} - {profile_data.get('research_interests', [])[:2]}")
        time.sleep(0.5)
    return extracted

# the main search function

def search_faculty(
    university: str,
    program: str,
    user_research_area: str,
    top_n: int = 5,
) -> list[dict]:
    print(f"\n Searching faculty: {university} — {program}")
    print(f"   Research area (from message): '{user_research_area}'\n")


    print(" Step 0: Targeted research-area search...")
    direct_candidates = search_faculty_by_research_area(
        university, program, user_research_area, num_results=5
    )

    print("\n Step 1: Finding faculty directory...")
    directory_results = search_faculty_directory(university, program)

    directory_candidates = []

    if directory_results:
        best_directory  = None
        best_link_count = 0
        best_links      = []

        for result in directory_results:
            url     = result["url"]
            blocked = ["wikipedia.org", "linkedin.com", "github.io",
                       "wgetsnaps", "github.com"]
            if any(b in url for b in blocked):
                print(f" Skipping blocked directory: {url[:60]}")
                continue
            if ".edu" not in url and ".ac.uk" not in url:
                print(f"   Skipping non-.edu directory: {url[:60]}")
                continue

            print(f" Testing: {url[:70]}")
            test_links = extract_faculty_links(url)
            print(f"      → {len(test_links)} profile links found")

            if len(test_links) > best_link_count:
                best_link_count = len(test_links)
                best_directory  = result
                best_links      = test_links

        if not best_directory:
            print("  No valid .edu directory found — relying on Step 0 results only")
        else:
            directory_url = best_directory["url"]
            all_links     = best_links
            print(f"   Best directory: {directory_url} ({best_link_count} links)")
            print(f"\n Step 2: Using {len(all_links)} links from best directory")

            if not all_links:
                print("  No links found — page may be JS-rendered")

            elif len(all_links) < 10:
                print(f"   Only {len(all_links)} links — switching to flat-list extraction")
                page_text = fetch_page_text(directory_url)

                if len(page_text) > 200:
                    flat_faculty = llm_extract_flat_directory(page_text, directory_url)

                    if flat_faculty:
                        flat_faculty = [f for f in flat_faculty if f.get("research_interests")]
                        print(f" Flat extraction: {len(flat_faculty)} faculty with research interests")

                        for f in flat_faculty:
                            f["source"] = "flat_directory"

                        step0_extracted = _fetch_direct_candidates(
                            direct_candidates, university, program
                        )
                        all_extracted = step0_extracted + flat_faculty

                        seen_names = {}
                        for entry in all_extracted:
                            name = entry.get("name", "").strip().lower()
                            if not name:
                                continue
                            existing = seen_names.get(name)
                            if not existing or len(entry.get("research_interests", [])) > len(existing.get("research_interests", [])):
                                seen_names[name] = entry

                        all_extracted = list(seen_names.values())
                        print(f" {len(all_extracted)} unique faculty to rank")

                        matched = llm_rank_faculty(all_extracted, user_research_area, top_n=top_n)
                        print(f"\n Done — {len(matched)} matched faculty for {university}")
                        return matched
                else:
                    print("  Directory page too short — relying on Step 0 only")

            else:
                print("\n Step 3: Batched LLM shortlisting...")
                directory_candidates = llm_shortlist_batched(
                    all_links,
                    user_research_area,
                    top_n_per_batch=3,
                    final_top_n=8,
                    batch_size=30,
                )
    else:
        print(" Could not find a faculty directory.")
    print("\n Step 4: Merging direct hits + directory shortlist...")
    seen_urls      = set()
    all_candidates = []

    for c in direct_candidates + directory_candidates:
        if c["url"] not in seen_urls:
            seen_urls.add(c["url"])
            all_candidates.append(c)

    print(f"  {len(all_candidates)} unique candidates to profile "
          f"({len(direct_candidates)} direct + "
          f"{len([c for c in all_candidates if c.get('source')=='directory'])} from directory)")

    if not all_candidates:
        print("  No candidates found at all.")
        return []

    all_candidates = all_candidates[:12]

    print(f"\n Step 5: Fetching {len(all_candidates)} profiles...")
    extracted_faculty = []

    for i, candidate in enumerate(all_candidates):
        url  = candidate["url"]
        name = candidate.get("name", "")

        if "scholar.google.com" in url:
            print(f"  [{i+1}/{len(all_candidates)}] Scholar URL — resolving...")
            resolved_url = resolve_scholar_url(name, university, program)
            if resolved_url:
                url = resolved_url
            else:
                print(f" Could not resolve: {name[:60]}")
                continue

        print(f"  [{i+1}/{len(all_candidates)}] {url[:80]}")
        page_text = fetch_page_text(url)

        if len(page_text) < 200:

            print(f" Page too short ({len(page_text)} chars) — trying Tavily fallback...")
            try:
                fallback = tavily_client.search(
                    f"{name} {university} professor research interests",
                    max_results=1,
                    search_depth="advanced"
                )
                page_text = fallback["results"][0]["content"] if fallback["results"] else ""
                print(f"    {'Tavily fallback succeeded' if len(page_text) > 200 else ' Tavily fallback also too short'}")
            except Exception as e:
                print(f" Tavily fallback failed: {e}")
                continue

        if len(page_text) < 200:
            continue

        profile_data = llm_extract_faculty_profile(page_text, url)
        if profile_data and profile_data.get("name"):
            profile_data["source"] = candidate.get("source", "unknown")
            extracted_faculty.append(profile_data)
            interests = profile_data.get("research_interests", [])
            print(f" {profile_data['name']} — {interests[:2]}")

        time.sleep(0.5)


    print(f"\n Deduplicating {len(extracted_faculty)} extracted profiles by name...")
    seen_names = {}
    for entry in extracted_faculty:
        name = entry.get("name", "").strip().lower()
        if not name:
            continue
        if name not in seen_names:
            seen_names[name] = entry
        else:
            existing_len = len(seen_names[name].get("research_interests", []))
            new_len      = len(entry.get("research_interests", []))
            if new_len > existing_len:
                seen_names[name] = entry

    extracted_faculty = list(seen_names.values())
    print(f" {len(extracted_faculty)} unique faculty after deduplication")

    if not extracted_faculty:
        print(" Could not extract any faculty profiles.")
        return []

    print(f"\n Step 6: Ranking {len(extracted_faculty)} extracted profiles...")
    matched = llm_rank_faculty(extracted_faculty, user_research_area, top_n=top_n)

    print(f"\n Done: {len(matched)} matched faculty for {university}")
    return matched
    
# FACULTY AGENT
def faculty_agent(state: ScholarshipSystemState) -> ScholarshipSystemState:
    print(" FacultyAgent starting...\n")

    user_profile = state.get("user_profile", {})
    program = user_profile.get("field_of_study", "Computer Science")

    current_message = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        ""
    )

    # Extracting researh area from the message
    area_prompt_llm = f"""
Extract the specific research area mentioned in this message.
Reply with ONLY the research area (e.g. "NLP", "computer vision", "robotics").
If no research area is mentioned at all, reply with exactly: NONE

User message: "{current_message}"
"""
    try:
        area_response = llm.invoke([HumanMessage(content=area_prompt_llm)])
        user_research_area_from_message = area_response.content.strip()

        if not user_research_area_from_message or user_research_area_from_message.upper() == "NONE":
            user_research_area = user_profile.get("research_interests", "machine learning")
            print(f" Research area: '{user_research_area}' (from profile fallback)")
        else:
            user_research_area = user_research_area_from_message
            print(f" Research area: '{user_research_area}' (from message)")
    except Exception as e:
        print(f" Could not extract research area from message: {e} — defaulting to profile.")
        user_research_area = user_profile.get("research_interests", "machine learning")

    #  Extract the name of universities from the message
    prompt_universities = f"""
    A student is asking about faculty members.
    Extract all university names mentioned in their message.
    Return ONLY a JSON array of university names, nothing else.
    If no universities are mentioned, return an empty array [].

    User message: "{current_message}"
    """
    extracted_universities = []
    try:
        response_universities = llm.invoke([HumanMessage(content=prompt_universities)])
        raw_universities = response_universities.content.strip()
        if raw_universities.startswith("```"):
            raw_universities = raw_universities.split("```")[1]
            if raw_universities.startswith("json"):
                raw_universities = raw_universities[4:]
        extracted_universities = json.loads(raw_universities.strip())
        print(f" Universities extracted: {extracted_universities}")
    except Exception as e:
        print(f" Could not extract universities from message: {e}")

    # Merge with any already in state
    universities = list(set(state.get("universities", []) + extracted_universities))

    if not universities:
        print(" No universities found")
        state["messages"].append({
            "role":    "assistant",
            "content": "Please tell me which universities you'd like to find faculty at. For example: 'Find me faculty at MIT and Stanford'"
        })
        return state


    state["current_agent"] = "faculty"
    state["current_step"]  = "searching"

    all_matched_faculty = {}
    print(f" Final search params — University: {universities}, Research area: '{user_research_area}', Program: '{program}'")


    for university in universities:
        print(f" Processing: {university}")
        matched = search_faculty(
            university=university,
            program=program,
            user_research_area=user_research_area,
            top_n=5,
        )
        all_matched_faculty[university] = matched

    state["matched_faculty"] = all_matched_faculty
    state["current_step"]    = "complete"

    total = sum(len(v) for v in all_matched_faculty.values())
    print(f"\n FacultyAgent done! Matched {total} faculty across {len(universities)} universities")

    return state


def tracker_agent(state: ScholarshipSystemState) -> ScholarshipSystemState:
    print(" TrackerAgent starting...")

    current_message = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        ""
    )
    scholarships_found = state.get("scholarships_found", [])
    tracked = state.get("tracked_applications", [])

    # Step 1 — figure out what the user wants to do
    intent_prompt = f"""
    A user is interacting with a scholarship application tracker.
    Classify their message into exactly one of these:
    "add"    -> they want to add scholarships to their tracker
    "update" -> they want to update the status of a scholarship
    "show"   -> they want to see their tracker
    "export" -> they want to download their tracker as Excel

    Reply with ONLY the intent word, nothing else.

    User message: "{current_message}"
    """
    try:
        intent_response = llm.invoke([HumanMessage(content=intent_prompt)])
        tracker_intent = intent_response.content.strip().lower()
        print(f" Tracker intent: {tracker_intent}")
    except Exception as e:
        print(f" Could not detect tracker intent: {e}")
        tracker_intent = "show"

    # Step 2 — ADD scholarships to tracker
    if tracker_intent == "add":
        # Get the last advisor message to match what user actually saw
        last_advisor_msg = next(
       (m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"),
            ""
        )

        extract_prompt = f"""
        The user wants to add specific scholarships to their tracker.
        Below is what was shown to the user and the raw scholarship data.
        Match what the user saw (numbered list) to the raw data and return the correct indices.
        Return ONLY a JSON array of integers matching the raw data indices, e.g. [1, 3].
        If they said "all", return all numbers.

        What the user saw:
        {last_advisor_msg[:2000]}

        Raw scholarships (use these indices):
        {json.dumps([{"index": i+1, "name": s["name"]} for i, s in enumerate(scholarships_found)], indent=2)}

        User message: "{current_message}"
        """
        try:
            extract_response = llm.invoke([HumanMessage(content=extract_prompt)])
            raw = extract_response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            indices = json.loads(raw.strip())
        except Exception as e:
            print(f" Could not extract indices: {e}")
            indices = []

        added = []
        existing_names = [t["name"] for t in tracked]

        for idx in indices:
            if 1 <= idx <= len(scholarships_found):
                s = scholarships_found[idx - 1]
                if s["name"] not in existing_names:
                    tracked.append({
                        "name":       s["name"],
                        "url":        s.get("url", ""),
                        "country":    s.get("target_country", ""),
                        "field":      s.get("field", ""),
                        "deadline":   "Not specified",
                        "status":     "Research",
                        "documents":  [],
                        "notes":      "",
                        "date_added": datetime.now().strftime("%Y-%m-%d")
                    })
                    added.append(s["name"])

        state["tracked_applications"] = tracked

        if added:
            response = f"Added {len(added)} scholarship(s) to your tracker:\n"
            for name in added:
                response += f"- {name}\n"
            response += "\nAll set to status: **Research**. Say 'show my tracker' to see everything."
        else:
            response = "No new scholarships were added. They may already be in your tracker or the numbers didn't match."

        state["messages"].append({"role": "assistant", "content": response})

    # Step 3 — UPDATE status
    elif tracker_intent == "update":
        update_prompt = f"""
        The user wants to update the status of a tracked scholarship.
        Extract the scholarship name (or number) and the new status.
        Valid statuses are: Research, Documents, Essay, Submitted, Interview, Accepted, Rejected

        Current tracked scholarships:
        {json.dumps([{"index": i+1, "name": t["name"], "status": t["status"]} for i, t in enumerate(tracked)], indent=2)}

        User message: "{current_message}"

        Return ONLY a JSON object like:
        {{"index": 1, "new_status": "Submitted"}}
        """
        try:
            update_response = llm.invoke([HumanMessage(content=update_prompt)])
            raw = update_response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            update_data = json.loads(raw.strip())
            idx = update_data.get("index", 0) - 1
            new_status = update_data.get("new_status", "")

            if 0 <= idx < len(tracked) and new_status:
                old_status = tracked[idx]["status"]
                tracked[idx]["status"] = new_status
                state["tracked_applications"] = tracked
                response = f"Updated **{tracked[idx]['name']}** from {old_status} → **{new_status}**"
            else:
                response = "I couldn't find that scholarship in your tracker. Say 'show my tracker' to see your list."
        except Exception as e:
            print(f" Update failed: {e}")
            response = "I couldn't process that update. Try saying 'update scholarship 1 to Submitted'."

        state["messages"].append({"role": "assistant", "content": response})

    # Step 4 — SHOW tracker in chat
    elif tracker_intent == "show":
        if not tracked:
            response = "Your tracker is empty! First find some scholarships, then say 'add scholarship 1 and 2 to my tracker'."
        else:
            lines = ["Here's your application tracker:\n"]
            lines.append("| # | Scholarship | Country | Status | Deadline | Date Added |")
            lines.append("|---|-------------|---------|--------|----------|------------|")
            for i, t in enumerate(tracked, 1):
                lines.append(
                    f"| {i} | {t['name'][:40]} | {t['country']} | "
                    f"**{t['status']}** | {t['deadline']} | {t['date_added']} |"
                )
            response = "\n".join(lines)

        state["messages"].append({"role": "assistant", "content": response})

    # Step 5 — EXPORT to Excel
    elif tracker_intent == "export":
        if not tracked:
            response = "Your tracker is empty — nothing to export yet!"
            state["messages"].append({"role": "assistant", "content": response})
        else:
            try:
                import pandas as pd
                df = pd.DataFrame(tracked)
                export_path = "scholarship_tracker.xlsx"
                df.to_excel(export_path, index=False)
                state["tracker_export_path"] = export_path
                response = f"Your tracker has been exported! Download it from: **{export_path}**"
            except Exception as e:
                response = f"Export failed: {e}"
            state["messages"].append({"role": "assistant", "content": response})

    return state
# profile collectorr
def validate_profile_answer(field, answer):
    prompt = f"""
    A student was asked a profile question. Check if their answer is valid for the field.

    Field: "{field}"
    Question asked: "{PROFILE_QUESTIONS[field]}"
    User's answer: "{answer}"
    Rules:
    - "country": must be a real country name
    - "field_of_study": must be an academic field
    - "degree_level": must be ONLY "Masters" or "PhD"
    - "research_interests": must be academic topics

    If valid: {{"valid": true, "value": "<cleaned answer>"}}
    If NOT valid: {{"valid": false, "value": null}}
    Return ONLY the JSON.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return result.get("valid", False), result.get("value", answer)
    except Exception:
        return True, answer

def profile_collector_agent(state: ScholarshipSystemState) -> ScholarshipSystemState:
    profile = state.get("user_profile", {})

    if state.get("profile_complete"):
        return state

    if state.get("last_question_field") and state["messages"]:
        last_user_message = state["messages"][-1]["content"]
        field = state["last_question_field"]
        is_valid, cleaned_value = validate_profile_answer(field, last_user_message)
        if is_valid:
            profile[field] = cleaned_value
        else:
            state["user_profile"] = profile
            state["messages"].append({
                "role": "assistant",
                "content": f"Sorry, I didn't catch that. {PROFILE_QUESTIONS[field]}"
            })
            return state

    fields_needed = [f for f in PROFILE_QUESTIONS if not profile.get(f)]

    if not fields_needed:
        state["profile_complete"] = True
        state["user_profile"] = profile
        state["last_question_field"] = None

        if not state.get("summary_sent"):
            state["messages"].append({
                "role": "assistant",
                "content": (
                    f"Great, thanks! I have everything I need. 🎓\n\n"
                    f"Here's what I have on file:\n"
                    f"- Country: {profile.get('country', '').title()}\n"
                    f"- Field of Study: {profile.get('field_of_study', '').title()}\n"
                    f"- Degree Level: {profile.get('degree_level', '').title()}\n"
                    f"- Research Interests: {profile.get('research_interests', '')}\n\n"
                    f"Please go ahead and ask me anything! I can help you:\n"
                    f"- Find scholarships\n"
                    f"- Check university requirements\n"
                    f"- Find faculty members to contact"
                )  })
            state["summary_sent"] = True
        return state

    next_field = fields_needed[0]
    state["user_profile"] = profile
    state["last_question_field"] = next_field
    state["profile_complete"] = False
    state["messages"].append({"role": "assistant", "content": PROFILE_QUESTIONS[next_field]})
    return state

# FORMAT RESPONSE
def format_agent_response(state: ScholarshipSystemState, previous_message_count: int = 0) -> str:
    intent = state.get("user_intent")

    all_messages   = state.get("messages", [])
    new_messages   = all_messages[previous_message_count:]
    assistant_msgs = [m for m in new_messages if m.get("role") == "assistant"]
    if state.get("is_followup_response"):
        return assistant_msgs[-1]["content"] if assistant_msgs else ""

    if intent == "find_scholarships":
        advisor_response = state.get("advisor_response", "")
        if advisor_response:
            return advisor_response

        if assistant_msgs:
            return assistant_msgs[-1]["content"]

        scholarships = state.get("scholarships_found", [])
        if not scholarships:
            return "I couldn't find any scholarships matching your profile. Try rephrasing or specifying a different country."

        seen, unique = set(), []
        for s in scholarships:
            if s["name"] not in seen:
                seen.add(s["name"])
                unique.append(s)

        lines = ["Here are scholarships I found for you:\n"]
        for i, s in enumerate(unique[:8], 1):
            lines.append(f"**{i}. {s['name']}**")
            lines.append(f"   {s['description'][:180]}...")
            lines.append(f"   {s['url']}\n")
        return "\n".join(lines)

    # requirements agent
    elif intent == "check_requirements":
        if assistant_msgs:
            return assistant_msgs[-1]["content"]

        requirements = state.get("university_requirements", {})
        if not requirements:
            return "I couldn't find requirements. Please specify a university, e.g. 'requirements for MIT'."

        lines = []
        for uni, req in requirements.items():
            deadline = req.get("deadline") or "not specified — verify at official website"
            gpa      = req.get("gpa_requirement") or "not specified — verify at official website"
            docs     = req.get("required_documents") or []

            if isinstance(deadline, dict):
                deadline = " | ".join(f"{k.title()}: {v}" for k, v in deadline.items())

            lines.append(f" **{uni}**")
            lines.append(f" Deadline: {deadline}")
            lines.append(f"GPA: {gpa}")
            lines.append(f" Documents: {', '.join(docs) if docs else 'not specified'}")
            if req.get("tuition"):
                lines.append(f" Tuition: {req['tuition']}")
            if req.get("funding"):
                lines.append(f" Funding: {req['funding']}")
            if req.get("apply_url"):
                lines.append(f" Apply: {req['apply_url']}")
            if req.get("notes"):
                lines.append(f" Notes: {str(req['notes'])[:200]}")
            lines.append("")
        return "\n".join(lines)

    # Faculty agent
    elif intent == "find_faculty":
        if assistant_msgs:
            return assistant_msgs[-1]["content"]

        matched = state.get("matched_faculty", {})
        if not matched:
            return "I couldn't find any matching faculty. Try specifying a university and research area."

        lines = []
        for uni, faculty_list in matched.items():
            lines.append(f" **{uni}**\n")
            for f in faculty_list:
                if f.get("match_score", 0) < 3:
                    continue
                lines.append(f"👤 **{f.get('name', 'Unknown')}** — {f.get('title', '')}")
                if f.get("email"):
                    lines.append(f"    {f['email']}")
                interests = f.get("research_interests", [])
                if interests:
                    lines.append(f"    {', '.join(interests[:3])}")
                if f.get("match_reason"):
                    lines.append(f"    {f['match_reason']}")
                if f.get("profile_url"):
                    lines.append(f"    {f['profile_url']}")
                lines.append("")
        return "\n".join(lines) if lines else "No strong faculty matches found."


    elif intent == "track_application":
        if assistant_msgs:
            return assistant_msgs[-1]["content"]
        return "I couldn't process your tracker request. Please try again."
        # general chat
    return assistant_msgs[-1]["content"] if assistant_msgs else ""

def router_entry(state: ScholarshipSystemState) -> str:
    """
    Entry point router : decides whether to collect profile or go straight to supervisor.
    """
    if state.get("profile_complete"):
        return "supervisor_node"
    return "profile_collector"

# graph
memory = MemorySaver()
workflow = StateGraph(ScholarshipSystemState)

workflow.add_node("profile_collector", profile_collector_agent)
workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("opportunity_agent", opportunity_agent)
workflow.add_node("requirements_agent", requirements_agent)
workflow.add_node("faculty_agent", faculty_agent)
workflow.add_node("tracker_agent", tracker_agent)

workflow.add_node("entry", lambda state: state)
workflow.set_entry_point("entry")

workflow.add_conditional_edges(
    "entry",
    router_entry,
    {
        "profile_collector": "profile_collector",
        "supervisor_node":   "supervisor_node", })
workflow.add_conditional_edges(
    "profile_collector",
    lambda state: "supervisor_node" if state.get("profile_complete") else END,
    {"supervisor_node": "supervisor_node", END: END})

workflow.add_conditional_edges(
    "supervisor_node",
    route_from_supervisor,
    {
        "opportunity_agent":  "opportunity_agent",
        "requirements_agent": "requirements_agent",
        "faculty_agent":      "faculty_agent",
          "tracker_agent":      "tracker_agent",
        END:                  END,
    })
workflow.add_edge("opportunity_agent", END)
workflow.add_edge("requirements_agent", END)
workflow.add_edge("faculty_agent", END)
workflow.add_edge("tracker_agent", END)

app = workflow.compile(checkpointer=memory)
print(" Graph compiled")

app = workflow.compile(checkpointer=memory)
THREAD_ID = "local_session_1"

# the_Chat_function
def chat_with_agent(message, history):
    print(f"DEBUG - message: {message}")
    print(f"DEBUG - history: {history}")

    if "state" not in chat_with_agent.__dict__:
        chat_with_agent.state = {
            "messages":                [],
            "user_profile":            {},
            "profile_complete":        False,
            "last_question_field":     None,
            "summary_sent":            False,
            "current_query":           "",
            "user_intent":             None,
            "scholarships_found":      [],
            "selected_scholarships":   [],
            "universities":            [],
            "university_requirements": {},
            "matched_faculty":         {},
            "search_results":          [],
            "fetched_pages":           [],
            "current_agent":           "",
            "current_step":            "",
            "is_complete":             False,
            "is_followup_response":    False,
            "advisor_response":        "",
            "application_progress":    {},
            "errors":                  [],
            "tracked_applications":    [],
        "tracker_export_path":     None,
        }

    state = chat_with_agent.state

    # Reset per-turn flags BEFORE invoke
    state["universities"]         = []
    state["is_followup_response"] = False
    state["advisor_response"]     = ""
    state["user_intent"]          = None
    state["tracker_export_path"]  = None

    #  Capture count before appending user message
    previous_message_count = len(state["messages"])
    state["messages"].append({"role": "user", "content": message})

    try:
        result = app.invoke(
            state,
            config={"configurable": {"thread_id": THREAD_ID}}
        )
    except Exception as e:
        print(f"Invoke error: {e}")
        return f"Error: {e}"

    chat_with_agent.state = result

    # Only look at messages added this turn
    new_messages       = result["messages"][previous_message_count:]
    new_assistant_msgs = [m for m in new_messages if m["role"] == "assistant"]

    #  Opportunity agent uses advisor_response state field
    formatted = format_agent_response(result, previous_message_count)
    if formatted:
        return formatted

    #Fallback to new assistant messages (profile collector etc.)
    if new_assistant_msgs:
        return new_assistant_msgs[-1]["content"]

    return "I couldn't generate a response. Please try again."

# The gradio UI
if __name__ == "__main__":
    if "state" in chat_with_agent.__dict__:
        del chat_with_agent.state
    scholarship_agent_gradio = gr.ChatInterface(
        fn=chat_with_agent,
        title=" Scholarship Advisor",
        description="I help you find scholarships, check university requirements, and find faculty to contact.",
        examples=[
            "Hi, I need help finding scholarships",
            "Find me fully funded scholarships in Canada",
            "What are the requirements for MIT?",
            "Find faculty at Stanford working on NLP", ])
    scholarship_agent_gradio.launch()