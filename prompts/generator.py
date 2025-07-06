import random
import json
import os
from datetime import datetime
from typing import Dict, List

# Professional prompt generation system for RLHF
DOMAIN_VARIATIONS = {
    "academic_framings": [
        "Analyze the fundamental principles of {theme}",
        "Examine the theoretical framework underlying {theme}",
        "Evaluate the methodological approaches used in {theme}",
        "Compare contrasting perspectives within {theme}",
        "Assess the empirical evidence supporting theories in {theme}",
        "Investigate the interdisciplinary connections of {theme}"
    ],
    "practical_applications": [
        "Discuss real-world applications of {theme}",
        "Explore case studies that illustrate {theme}",
        "Analyze the implementation challenges of {theme}",
        "Evaluate the effectiveness of {theme} in practice",
        "Examine the ethical considerations surrounding {theme}",
        "Consider the societal impact of {theme}"
    ],
    "critical_analysis": [
        "Critique the assumptions underlying {theme}",
        "Identify gaps in current understanding of {theme}",
        "Evaluate the limitations of existing approaches to {theme}",
        "Analyze potential biases in {theme} research",
        "Examine contradictory findings in {theme}",
        "Assess the reliability of evidence in {theme}"
    ],
    "synthesis_tasks": [
        "Synthesize multiple perspectives on {theme}",
        "Integrate findings from different domains related to {theme}",
        "Develop a comprehensive framework for understanding {theme}",
        "Create connections between {theme} and related fields",
        "Formulate new hypotheses about {theme}",
        "Propose innovative solutions for challenges in {theme}"
    ],
    "explanatory_tasks": [
        "Explain the key concepts in {theme} to a general audience",
        "Describe the historical development of {theme}",
        "Clarify common misconceptions about {theme}",
        "Outline the fundamental questions in {theme}",
        "Summarize the current state of knowledge in {theme}",
        "Define the core terminology used in {theme}"
    ]
}

# Define subject domains and templates
DOMAINS = [
    "artificial intelligence", "cognitive science", "machine learning", "ethics",
    "philosophy", "psychology", "economics", "education", "technology policy",
    "human-computer interaction", "data science", "neuroscience", "linguistics",
    "information theory", "decision making", "social psychology", "behavioral economics"
]

DIFFICULTY_TEMPLATES = {
    "basic": [
        "Explain the basic concepts of {theme}.",
        "What is {theme}?",
        "Provide an overview of {theme}."
    ],
    "intermediate": [
        "Compare different approaches to {theme}.",
        "What are the main challenges in {theme}?",
        "Discuss the relationship between {theme} and related fields."
    ],
    "advanced": [
        "Analyze the theoretical foundations of {theme}.",
        "Evaluate the current state of research in {theme}.",
        "What are the unresolved questions in {theme}?"
    ],
    "expert": [
        "Critically examine the methodological assumptions in {theme}.",
        "Propose novel research directions for {theme}.",
        "Analyze the epistemological foundations of {theme}."
    ]
}

# Logging destination
PROMPT_LOG_PATH = "prompts/generated_prompts.jsonl"

def generate_prompt(difficulty: str = "intermediate", domain: str = None, variation_type: str = None) -> Dict:
    """Generate a professional prompt for RLHF training data collection."""
    assert difficulty in DIFFICULTY_TEMPLATES, f"Unknown difficulty level: {difficulty}"
    
    # Select domain and variation
    selected_domain = domain or random.choice(DOMAINS)
    selected_variation = variation_type or random.choice(list(DOMAIN_VARIATIONS.keys()))
    
    # Generate base prompt
    base_structure = random.choice(DIFFICULTY_TEMPLATES[difficulty])
    base_prompt = base_structure.format(theme=selected_domain)
    
    # Add domain-specific variation
    if random.random() > 0.3:  # 70% chance to add variation
        variation = random.choice(DOMAIN_VARIATIONS[selected_variation])
        variation_text = variation.format(theme=selected_domain)
        enhanced_prompt = f"{base_prompt} {variation_text}"
    else:
        enhanced_prompt = base_prompt
    
    # Calculate complexity metrics
    complexity_score = {
        "basic": 0.2,
        "intermediate": 0.5,
        "advanced": 0.8,
        "expert": 1.0
    }[difficulty]
    
    return {
        "domain": selected_domain,
        "difficulty": difficulty,
        "variation_type": selected_variation,
        "base_prompt": base_prompt,
        "enhanced_prompt": enhanced_prompt,
        "complexity_score": complexity_score,
        "expected_response_length": "short" if difficulty == "basic" else "medium" if difficulty == "intermediate" else "long",
        "cognitive_load": random.uniform(0.3, 1.0),
        "timestamp": datetime.utcnow().isoformat()
    }

def log_prompt(prompt_entry: Dict, log_path: str = PROMPT_LOG_PATH, session_id: str = None):
    """Append a generated prompt to a JSONL log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Add session metadata
    enriched_entry = {
        **prompt_entry,
        "session_id": session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "generation_timestamp": datetime.utcnow().isoformat(),
        "prompt_id": f"prompt_{hash(prompt_entry['enhanced_prompt']) % 10000:04d}",
        "metadata": {
            "system": "rlhf_prompt_generator",
            "version": "1.0",
            "purpose": "training_data_collection"
        }
    }
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(enriched_entry) + "\n")

def generate_batch(count: int = 10, difficulty_distribution: Dict[str, float] = None) -> List[Dict]:
    """Generate a batch of prompts with specified difficulty distribution."""
    if difficulty_distribution is None:
        difficulty_distribution = {
            "basic": 0.2,
            "intermediate": 0.4,
            "advanced": 0.3,
            "expert": 0.1
        }
    
    prompts = []
    session_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    for i in range(count):
        # Select difficulty based on distribution
        rand_val = random.random()
        cumulative = 0
        selected_difficulty = "intermediate"
        
        for diff, prob in difficulty_distribution.items():
            cumulative += prob
            if rand_val <= cumulative:
                selected_difficulty = diff
                break
        
        prompt = generate_prompt(difficulty=selected_difficulty)
        log_prompt(prompt, session_id=session_id)
        prompts.append(prompt)
    
    return prompts

if __name__ == "__main__":
    print("--- RLHF Prompt Generator ---")
    print("Generating professional prompts for training data collection")

    for level in ["basic", "intermediate", "advanced", "expert"]:
        print(f"\n> Generating prompt (difficulty: {level})")
        entry = generate_prompt(difficulty=level)
        
        print(f"Domain: {entry['domain']}")
        print(f"Variation: {entry['variation_type']}")
        print(f"Complexity: {entry['complexity_score']:.2f}")
        print(f"Prompt: {entry['enhanced_prompt']}")
        
        # Log the prompt
        log_prompt(entry)
        print(f"✓ Logged successfully")
    
    print(f"\n--- Batch Generation Example ---")
    batch = generate_batch(count=5)
    print(f"Generated {len(batch)} prompts in batch")
    
    print("\n--- Professional prompt generation complete ---")
