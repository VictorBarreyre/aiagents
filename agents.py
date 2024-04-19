from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

# Initializing the language model
mixtral = Ollama(model="mixtral")

# Define the topic of research
topic = "web development"

# Define the research agent
research_agent = Agent(
    role="Researcher",
    goal=f"From your memory, gather relevant information about how an expert at {topic} would approach a project",
    backstory=f"You are an AI assistant that extracts relevant information for {topic} experts from your knowledge base",
    verbose=True,
    allow_delegation=False,
    llm=mixtral
)

# Define the prompt engineer agent
prompt_agent = Agent(
    role="Prompt Engineer",
    goal=f"Write a single structured prompt in markdown explaining how a world-class {topic} expert would approach a project",
    backstory=f"You are an AI assistant that writes a single prompt explaining how {topic} experts do it", 
    verbose=True,
    allow_delegation=False,
    llm=mixtral
)

# Define tasks for each agent
gather_info = Task(
    description=f"From your knowledge base, collect relevant information about {topic} experts",
    agent=research_agent,
    expected_output=f"A clear list of key points related to {topic} experts and how they work"
)

write_prompt = Task(
    description=f"Write a single structured prompt in markdown explaining how a world-class {topic} expert would approach a project",
    agent=prompt_agent,
    expected_output=f"Single structured prompt in markdown explaining how a world-class {topic} expert would approach a project"
)

# Define the crew
crew = Crew(
    agents=[research_agent, prompt_agent],
    tasks=[gather_info, write_prompt],
    verbose=True,
    process=Process.sequential
)

# Execute the crew tasks and print output
try:
    output = crew.kickoff()
    print(output)
except Exception as e:
    print(f"An error occurred: {e}")
