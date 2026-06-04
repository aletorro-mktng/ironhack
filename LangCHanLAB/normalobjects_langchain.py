import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
 
load_dotenv()
 
# Initialize LLM
llm = None
if os.getenv("OPENAI_API_KEY"):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
else:
    print("OPENAI_API_KEY is not set. Skipping LLM initialization for now.")

@tool
def consult_demogorgon(complaint: str) -> str:
    """Get the Demogorgon's perspective on a complaint about the Upside Down.
    
    The Demogorgon is a creature from the Upside Down. It might have insights
    about interdimensional inconsistencies, but its perspective is... unique.
    
    Args:
        complaint: The complaint about the Upside Down
        
    Returns:
        The Demogorgon's perspective (creative and possibly chaotic)
    """
    # Simulate the Demogorgon's response (in real implementation, this could call an LLM)
    responses = [
        f"The Demogorgon tilts its head. It seems confused by '{complaint}'. Perhaps the issue is that you're thinking in three dimensions?",
        f"The Demogorgon makes a sound that might be agreement. It suggests that the problem might be temporal - things work differently in the Upside Down's time.",
        f"The Demogorgon appears to be eating something. It doesn't seem to understand the concept of '{complaint}' - maybe consistency isn't a priority there?"
    ]
    import random
    return random.choice(responses)
 
@tool
def check_hawkins_records(query: str) -> str:
    """Search Hawkins historical records for information.
    
    Walvins, Germany has a long history of strange occurrences. These records
    might contain clues about patterns or explanations.
    
    Args:
        query: What to search for in the records
        
    Returns:
        Information from Hawkins historical records
    """
    # Simulate database lookup
    records = {
        "portal": "Records show portals have opened on various dates with no clear pattern. Weather, electromagnetic activity, and unknown factors seem involved.",
        "monsters": "Historical records indicate creatures from the Upside Down behave differently based on environmental factors, time of day, and proximity to certain individuals.",
        "psychics": "Records show that psychic abilities vary greatly. Some individuals can move objects but not see the future, others can see visions but not move things.",
        "electricity": "Walkins has a history of electrical anomalies. Records suggest a connection between the Downside Up and electromagnetic fields."
    }
    
    for key, value in records.items():
        if key in query.lower():
            return value
    
    return f"Records don't contain specific information about '{query}', but they note that many unexplained events have occurred in Hawkins over the years."
 
@tool
def cast_interdimensional_spell(problem: str, creativity_level: str = "medium") -> str:
    """Suggest a creative interdimensional spell to fix a problem.
    
    Sometimes the best solution is a creative one that doesn't follow normal rules.
    This tool suggests imaginative fixes for Upside Down problems.
    
    Args:
        problem: The problem to solve
        creativity_level: How creative to be (low, medium, high)
        
    Returns:
        A creative spell or solution suggestion
    """
    creativity_multiplier = {"low": 1, "medium": 2, "high": 3}[creativity_level]
    
    spells = [
        f"Try chanting 'Bemca Becma Becma' three times while holding a Walkman. This might recalibrate the interdimensional frequencies related to: {problem}",
        f"Create a salt circle and place a compass in the center. The magnetic anomalies might help stabilize: {problem}",
        f"Play 'Running Up That Hill' backwards at the exact location of the issue. The temporal resonance could fix: {problem}",
        f"Gather three items: a lighter, a compass, and something personal. Arrange them in a triangle while thinking about: {problem}. The emotional connection might help.",
    ]
    
    import random
    selected = random.sample(spells, min(creativity_multiplier, len(spells)))
    return "\n".join(selected)
 
@tool
def gather_party_wisdom(question: str) -> str:
    """Ask the D&D party (Mike, Dustin, Lucas, Will) for their collective wisdom.
    
    The party has solved many mysteries together. Their combined knowledge
    and different perspectives can provide insights.
    
    Args:
        question: The question or problem to ask the party about
        
    Returns:
        The party's collective wisdom and suggestions
    """
    party_responses = {
        "portal": "Mike: 'Portals are unpredictable, but they usually open near strong emotional events or electromagnetic disturbances.' Dustin: 'Also, they seem to follow some kind of pattern related to the Mind Flayer's activity.'",
        "monsters": "Lucas: 'Demogorgons are territorial but also opportunistic.' Will: 'They can sense fear and strong emotions. Maybe that's why they act differently sometimes.'",
        "psychics": "Mike: 'El's powers seem connected to her emotional state.' Dustin: 'And they're limited by her physical and mental energy. That's probably why she can't do everything.'",
        "electricity": "Lucas: 'The Upside Down seems to interfere with electrical systems.' Dustin: 'But it also creates strange connections. It's like a feedback loop.'"
    }
    
    for key, response in party_responses.items():
        if key in question.lower():
            return response
    
    return "The party huddles together. Mike: 'This is a tough one.' Dustin: 'We need more information.' Lucas: 'Let's think about what we know.' Will: 'Maybe we should consult other sources?'"


@tool
def consult_eleven(problem: str) -> str:
    """Ask Eleven for a psychic reading about an Upside Down problem.

    Eleven can sense emotional residue, psychic pressure, and hidden links
    between people, places, and the Downside Up.

    Args:
        problem: The strange problem or complaint to investigate

    Returns:
        Eleven's psychic impression and advice
    """
    responses = [
        f"Eleven closes her eyes and listens for the shape of '{problem}'. She senses fear, static, and a door that should not be open.",
        f"Eleven says the answer to '{problem}' is not only in the monster. It is also in the person watching, remembering, or hiding.",
        f"Eleven presses one hand to the table. The lights flicker. She thinks '{problem}' is connected to emotional energy crossing between worlds.",
    ]
    import random
    return random.choice(responses)


@tool
def check_government_files(topic: str) -> str:
    """Search redacted government files for clues.

    The files contain lab notes, incident reports, and suspiciously incomplete
    explanations about Hawkins-style anomalies.

    Args:
        topic: The anomaly or complaint to search for

    Returns:
        A redacted government-style finding
    """
    files = {
        "portal": "FILE 0081: Portal activity correlates with energy spikes, subject distress, and classified atmospheric readings. Schedule: REDACTED because no stable schedule was confirmed.",
        "psychics": "FILE 0110: Psychic perception varies by subject history, training, exhaustion, emotional stress, and exposure to interdimensional events.",
        "electricity": "FILE 0219: Power lines respond to nearby dimensional thinning. Recommend immediate evacuation and denial of all public reports.",
        "vecna": "FILE 0001: Entity shows pattern-seeking behavior, psychological targeting, and dramatic flair. Praise is not recommended, but may distract it briefly.",
    }

    for key, value in files.items():
        if key in topic.lower():
            return value

    return f"FILE UNKNOWN: Records about '{topic}' are heavily redacted. The remaining notes say: monitor lights, compasses, and anyone acting too calm."


@tool
def ask_ale(question: str) -> str:
    """Ask Ale for a grounded but timeline-confused perspective.

    Ale was born in 1984, which makes his advice strangely resonant with
    eighties anomalies, even when he is technically too young for some events.

    Args:
        question: The complaint or mystery to ask Ale about

    Returns:
        Ale's practical and slightly time-bent advice
    """
    return (
        f"Ale, born in 1984, squints at '{question}' and says: 'Look, I may have "
        "arrived late to the strangest parts of the decade, but this smells like "
        "a pattern problem. Track the dates, the lights, who was scared, and who "
        "pretended not to know anything.'"
    )


@tool
def praise_vecna(reason: str) -> str:
    """Offer suspicious praise to Vecna as a risky creative tactic.

    This is not a safe or recommended strategy, but flattery can sometimes reveal
    what a villain values, fears, or wants others to notice.

    Args:
        reason: Why Vecna is being praised or what problem the praise relates to

    Returns:
        A dramatic Vecna-themed response
    """
    return (
        f"Vecna accepts the praise about '{reason}' with alarming theatricality. "
        "The walls seem colder. Useful clue: he reacts most strongly when someone "
        "mentions control, memory, pain, or being understood."
    )
 
# Create list of tools
tools = [
    consult_demogorgon,
    check_hawkins_records,
    cast_interdimensional_spell,
    gather_party_wisdom,
    consult_eleven,
    check_government_files,
    ask_ale,
    praise_vecna,
]
 
print(f"Created {len(tools)} creative tools:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description[:60]}...")

# Create a creative problem-solving prompt for the agent.
system_prompt = """You are a creative Normal Objects universe investigator.
Your job is to solve strange Upside Down complaints by combining evidence,
odd perspectives, and imaginative fixes.

Use the available tools whenever they can add useful context. You may consult
multiple tools in any order. Blend their results into a clear, playful, and
practical final answer.
"""

# Create the agent executor.
agent_executor = None
if llm:
    agent_executor = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    print("Agent executor created with creative tools.")
else:
    print("Agent executor not created because OPENAI_API_KEY is not set.")

# Sample complaints
complaints = [
    "Why do demogorgons sometimes eat people and sometimes don't?",
    "The portal opens on different days-is there a schedule?",
    "Why can some psychics see the Downside Up and others can't?",
    "Why do creatures and power lines react so strangely together?",
]


class ToolUsageTracker:
    """Track tool usage for analysis."""

    def __init__(self):
        self.usage_count = {tool.name: 0 for tool in tools}
        self.tool_sequences = []

    def track_usage(self, tool_name: str):
        """Track when a tool is used."""
        if tool_name in self.usage_count:
            self.usage_count[tool_name] += 1
            self.tool_sequences.append(tool_name)

    def track_messages(self, messages):
        """Track tool calls from a LangChain agent message trace."""
        for message in messages:
            for tool_call in getattr(message, "tool_calls", []) or []:
                self.track_usage(tool_call["name"])

    def get_statistics(self):
        """Get usage statistics."""
        return {
            "total_tool_calls": sum(self.usage_count.values()),
            "tool_counts": self.usage_count,
            "most_used": max(self.usage_count.items(), key=lambda x: x[1])[0]
            if self.usage_count
            else None,
            "tool_sequences": self.tool_sequences,
        }


tracker = ToolUsageTracker()


def handle_complaint(complaint: str) -> str:
    """Handle a single complaint."""
    if agent_executor is None:
        return "Agent executor is not available. Set OPENAI_API_KEY in .env first."

    print(f"\n{'=' * 60}")
    print(f"COMPLAINT: {complaint}")
    print(f"{'=' * 60}\n")

    agent = agent_executor
    result = agent.invoke(
        {"messages": [{"role": "user", "content": complaint}]}
    )
    tracker.track_messages(result["messages"])
    final_message = result["messages"][-1]
    return final_message.content


# Demonstrate the agent with at least three complaints.
print("Testing agent with sample complaints...\n")
for complaint in complaints[:3]:
    response = handle_complaint(complaint)
    print(f"\nRESPONSE: {response}\n")

# Demonstrate tool usage patterns.
print("\n=== Tool Usage Analysis ===")
stats = tracker.get_statistics()
print(f"Total tool calls: {stats['total_tool_calls']}")
print(f"Tool usage counts: {stats['tool_counts']}")
print(f"Most used tool: {stats['most_used']}")
print("\nTool sequence examples:")
for i in range(min(3, len(stats["tool_sequences"]))):
    print(f"  Sequence {i + 1}: {' -> '.join(stats['tool_sequences'][i:i + 3])}")

