from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class TokenUsage:
    """Stores token usage details for a generation event."""
    prompt: Optional[int] = None
    completion: Optional[int] = None
    total: Optional[int] = None

@dataclass
class ModelGenerationMetadata:
    """Optional metadata about the model and parameters used for generating completions."""
    model: str  # Assuming model ID should always be present if metadata is logged
    temperature: float # Assuming temperature should always be present
    top_p: float       # Assuming top_p should always be present
    tokens: Optional[TokenUsage] = None # Token usage itself might be missing from source
    estimated_cost: Optional[float] = None

@dataclass
class VotePredictionExample:
    """
    Represents a single training example for a vote prediction model.
    """
    prompt: str
    completions: List[str]
    chosen_index: int
    confidence: Optional[float] = None
    annotation: Optional[str] = None
    generation_metadata: Optional[ModelGenerationMetadata] = None

    def to_dict(self):
        return asdict(self)

if __name__ == "__main__":
    import json # Needed for json.dumps

    print("--- Running dataset_schema.py __main__ block (Hybrid Version) ---")
    
    # Example where token usage might be present
    sample_tokens_present = TokenUsage(prompt=50, completion=300, total=350)
    sample_gen_metadata_full = ModelGenerationMetadata(
        model="deepseek-chat-v1",
        temperature=0.7,
        top_p=0.9,
        tokens=sample_tokens_present,
        estimated_cost=0.00015
    )
    example_instance_full_meta = VotePredictionExample(
        prompt="What is the future of AI in education?",
        completions=[
            "AI will personalize learning paths for every student.",
            "AI will automate grading and administrative tasks for teachers."
        ],
        chosen_index=0,
        confidence=0.85,
        annotation="Personalization is key.",
        generation_metadata=sample_gen_metadata_full
    )
    print("\n--- Sample with Full Generation Metadata (as_dict) ---")
    print(json.dumps(example_instance_full_meta.to_dict(), indent=2))

    # Example where token usage might be missing, but other metadata is present
    sample_gen_metadata_no_tokens = ModelGenerationMetadata(
        model="deepseek-chat-v2",
        temperature=0.6,
        top_p=0.95,
        # tokens field is omitted, relying on its Optional default
        estimated_cost=0.00010
    )
    example_instance_partial_meta = VotePredictionExample(
        prompt="Tell me a joke about Python.",
        completions=["Why was the Python developer so calm? Because he had inner peace (and good error handling).", "Snakes on a plane!"],
        chosen_index=0,
        generation_metadata=sample_gen_metadata_no_tokens
    )
    print("\n--- Sample with Partial Generation Metadata (missing tokens, as_dict) ---")
    print(json.dumps(example_instance_partial_meta.to_dict(), indent=2))
    
    # Example with no generation metadata at all
    example_minimal = VotePredictionExample(
        prompt="Summarize 'Hamlet'.",
        completions=["Revenge tragedy.", "It's complicated."],
        chosen_index=0
    )
    print("\n--- Sample Minimal (no generation_metadata, as_dict) ---")
    print(json.dumps(example_minimal.to_dict(), indent=2))
    print("\n--- dataset_schema.py __main__ block finished ---")