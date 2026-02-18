from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class SearchState(TypedDict):
    user_profile : dict
    queries : list[str]
    raw_jobs : list[dict]
    scored_jobs: list[dict]

def User_input(state: SearchState):
    user_profile = state.get("user_profile", {})
    name = user_profile.get("name", "")
    skills = user_profile.get("skills", [])
    preferred_role = user_profile.get("preferred_role", "")
    location = user_profile.get("location", "")

    if isinstance(skills, str):
        skills = [skill.strip() for skill in skills.split(",")]

    state["user_profile"] = {
        "name": name,
        "skills": skills,
        "preferred_role": preferred_role,
        "location": location
    }

    return state



def Query_builder(state: SearchState):
    skills = state["user_profile"]["skills"]
    role = state["user_profile"]["preferred_role"]

    queries = [role]

    for skill in skills:
        queries.append(f"{role} {skill}")

    state["queries"] = list(set(queries))

    return state

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

def Linkedin_job_scrapper(state):

    location = state["user_profile"]["location"]
    queries = state["queries"]

    options = Options()
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)

    all_jobs = []

    for query in queries:
        query_encoded = query.replace(" ", "%20")
        location_encoded = location.replace(" ", "%20")

        # 🔥 Past week filter (faster + latest jobs)
        url = f"https://www.linkedin.com/jobs/search/?keywords={query_encoded}&location={location_encoded}&f_TPR=r604800"
        
        driver.get(url)
        time.sleep(5)  # wait for page load

        # Small scroll to load more jobs
        driver.execute_script("window.scrollTo(0, 800);")
        time.sleep(2)

        job_cards = driver.find_elements(By.CLASS_NAME, "base-card")

        # Limit to first 10 jobs (speed optimization)
        job_cards = job_cards[:20]

        for card in job_cards:
            try:
                # Basic info
                title = card.find_element(By.CLASS_NAME, "base-search-card__title").text
                company = card.find_element(By.CLASS_NAME, "base-search-card__subtitle").text
                loc = card.find_element(By.CLASS_NAME, "job-search-card__location").text
                link = card.find_element(By.TAG_NAME, "a").get_attribute("href")

                # Click card to load description panel
                card.click()
                time.sleep(2)

                # Extract description
                try:
                    description = driver.find_element(
                        By.CLASS_NAME, "show-more-less-html__markup"
                    ).text
                except:
                    description = ""

                all_jobs.append({
                    "title": title,
                    "company": company,
                    "location": loc,
                    "link": link,
                    "description": description
                })

            except:
                continue

    driver.quit()

    state["raw_jobs"] = all_jobs
    return state

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def match_scoring_node(state):

    raw_jobs = state["raw_jobs"]
    user_profile = state["user_profile"]

    # Convert user skills to text
    user_text = " ".join(user_profile["skills"])

    # Generate user embedding
    user_vector = embedding_model.encode([user_text])

    scored_jobs = []

    for job in raw_jobs:

        # Use description for AI matching
        job_text = job.get("description", "")

        # Fallback if description is empty (VERY IMPORTANT)
        if not job_text.strip():
            job_text = job["title"] + " " + job["company"] + " " + job["location"]

        # Generate job embedding
        job_vector = embedding_model.encode([job_text])

        # Cosine similarity score
        score = cosine_similarity(user_vector, job_vector)[0][0]

        job_with_score = job.copy()
        job_with_score["score"] = float(score)

        scored_jobs.append(job_with_score)

    # Sort by best match first
    scored_jobs = sorted(scored_jobs, key=lambda x: x["score"], reverse=True)

    state["scored_jobs"] = scored_jobs
    return state


graph = StateGraph(SearchState)

graph.add_node('User_input',User_input)
graph.add_node('Query_builder',Query_builder)
graph.add_node('Linkedin_job_scrapper',Linkedin_job_scrapper)
graph.add_node('match_scoring_node',match_scoring_node)

graph.add_edge(START,'User_input')
graph.add_edge('User_input','Query_builder')
graph.add_edge('Query_builder','Linkedin_job_scrapper')
graph.add_edge('Linkedin_job_scrapper','match_scoring_node')
graph.add_edge('match_scoring_node',END)

workflow = graph.compile()

