#!/usr/bin/env python3
"""
Rich Demo Data Generator for RLHF Dashboard

This script generates comprehensive, realistic demo data that simulates months
of intensive RLHF system usage. Creates rich patterns showing:
- Model evolution and improvement over time (60% → 85%+ accuracy)
- Diverse prompt domains and complexity levels
- Realistic usage patterns and temporal trends
- Comprehensive calibration and confidence data
- Rich metadata and reflection data
- Support for all dashboard visualizations

Perfect for portfolio showcases, demos, and feature exploration.
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any
import math

# Set random seeds for reproducible data
random.seed(42)
np.random.seed(42)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

class RichDemoDataGenerator:
    """Generates comprehensive, realistic RLHF demo data"""
    
    def __init__(self):
        self.start_date = datetime(2024, 8, 1)  # 6 months ago
        self.end_date = datetime(2025, 1, 8)   # Today
        self.total_days = (self.end_date - self.start_date).days
        
        # Model evolution parameters
        self.initial_accuracy = 0.58
        self.final_accuracy = 0.87
        self.initial_confidence_error = 0.25
        self.final_confidence_error = 0.08
        
        # Usage patterns
        self.total_prompts = 450  # Much richer dataset
        self.daily_variation = 0.4  # 40% variation in daily usage
        
        # Initialize prompt libraries
        self._init_prompt_libraries()
        
    def _init_prompt_libraries(self):
        """Initialize comprehensive prompt libraries by domain and complexity"""
        
        # Programming prompts (technical, practical)
        self.programming_prompts = [
            ("Write a Python function to implement binary search", "algorithm", "medium"),
            ("Debug this React component that won't render properly", "debugging", "hard"),
            ("Create a REST API endpoint for user authentication", "web_dev", "medium"),
            ("Optimize this SQL query for better performance", "database", "hard"),
            ("Implement a thread-safe cache in Java", "concurrency", "hard"),
            ("Write a Dockerfile for a Node.js application", "devops", "medium"),
            ("Create a Git pre-commit hook for code formatting", "tools", "medium"),
            ("Implement rate limiting middleware for Express.js", "backend", "hard"),
            ("Write unit tests for this authentication service", "testing", "medium"),
            ("Design a database schema for an e-commerce platform", "architecture", "hard"),
            ("Create a CI/CD pipeline using GitHub Actions", "devops", "medium"),
            ("Implement a recursive descent parser", "algorithms", "hard"),
            ("Write a Python decorator for function timing", "python", "medium"),
            ("Create a responsive CSS grid layout", "frontend", "easy"),
            ("Implement OAuth 2.0 authentication flow", "security", "hard"),
            ("Write a script to monitor server health", "automation", "medium"),
            ("Design a microservices architecture", "architecture", "hard"),
            ("Create a real-time chat application", "web_dev", "hard"),
            ("Implement data validation middleware", "backend", "medium"),
            ("Write a function to parse CSV files efficiently", "data_processing", "medium"),
        ]
        
        # AI/ML concepts and explanations  
        self.ai_ml_prompts = [
            ("Explain neural network backpropagation to a beginner", "education", "medium"),
            ("What are the ethical implications of facial recognition?", "ethics", "hard"),
            ("Compare supervised vs unsupervised learning", "concepts", "easy"),
            ("How do large language models generate text?", "concepts", "medium"),
            ("Explain the bias-variance tradeoff in machine learning", "theory", "medium"),
            ("What is the alignment problem in AI safety?", "safety", "hard"),
            ("How does transfer learning work in deep learning?", "concepts", "medium"),
            ("Explain the difference between AI, ML, and Deep Learning", "education", "easy"),
            ("What are the risks of AI-generated misinformation?", "ethics", "hard"),
            ("How do recommendation systems work?", "applications", "medium"),
            ("Explain overfitting and how to prevent it", "theory", "medium"),
            ("What is federated learning and why is it important?", "concepts", "hard"),
            ("How do generative adversarial networks (GANs) work?", "concepts", "hard"),
            ("Explain the transformer architecture", "architecture", "hard"),
            ("What are the limitations of current AI systems?", "theory", "medium"),
            ("How does reinforcement learning differ from other ML?", "concepts", "medium"),
            ("Explain the concept of AI consciousness", "philosophy", "hard"),
            ("What is few-shot learning in language models?", "concepts", "medium"),
            ("How do we measure AI system performance?", "evaluation", "medium"),
            ("What is the future of artificial general intelligence?", "future", "hard"),
        ]
        
        # Ethics, philosophy, and complex reasoning
        self.ethics_prompts = [
            ("Should AI development be regulated by governments?", "regulation", "hard"),
            ("Is it ethical to use AI for hiring decisions?", "workplace", "hard"),
            ("How do we ensure AI systems are fair and unbiased?", "fairness", "hard"),
            ("Should social media platforms control misinformation?", "social_media", "hard"),
            ("What are the implications of AI replacing human jobs?", "economics", "hard"),
            ("Is privacy dead in the digital age?", "privacy", "medium"),
            ("Should we develop autonomous weapons systems?", "military", "hard"),
            ("How do we balance innovation with AI safety?", "policy", "hard"),
            ("Is AI creativity genuine or just sophisticated mimicry?", "philosophy", "hard"),
            ("Should AI systems have rights?", "philosophy", "hard"),
            ("How do we maintain human agency in an AI world?", "philosophy", "hard"),
            ("What ethical frameworks should guide AI development?", "ethics", "hard"),
            ("Should there be a universal basic income for AI displacement?", "economics", "hard"),
            ("How do we preserve human dignity in automation?", "philosophy", "medium"),
            ("What are the environmental costs of large AI models?", "environment", "medium"),
        ]
        
        # Creative writing and storytelling
        self.creative_prompts = [
            ("Write a story about AI discovering emotions", "fiction", "medium"),
            ("Create a dialogue between a human and sentient AI", "dialogue", "medium"),
            ("Write a poem about the beauty of code", "poetry", "easy"),
            ("Describe a world where AI governs society", "worldbuilding", "hard"),
            ("Write a thriller about AI gone rogue", "fiction", "medium"),
            ("Create a children's story explaining how computers think", "education", "medium"),
            ("Write a monologue from the perspective of an algorithm", "creative", "hard"),
            ("Describe the emotional journey of learning to code", "personal", "easy"),
            ("Write a love letter from one AI to another", "romance", "medium"),
            ("Create a comedy sketch about tech support", "humor", "easy"),
        ]
        
        # Career and workplace advice
        self.career_prompts = [
            ("How to transition from junior to senior developer?", "career", "medium"),
            ("Best practices for remote work productivity", "workplace", "easy"),
            ("How to handle technical interviews effectively?", "interviews", "medium"),
            ("Building a strong engineering team culture", "management", "hard"),
            ("Dealing with imposter syndrome in tech", "personal", "medium"),
            ("How to stay current with rapidly changing technology?", "learning", "medium"),
            ("Negotiating salary as a software engineer", "career", "medium"),
            ("Building a personal brand as a developer", "career", "medium"),
            ("How to give effective code reviews?", "collaboration", "medium"),
            ("Managing technical debt in legacy systems", "engineering", "hard"),
        ]
        
        # Current events and analysis
        self.current_events_prompts = [
            ("Impact of climate change on technology infrastructure", "climate", "hard"),
            ("How will quantum computing affect cybersecurity?", "quantum", "hard"),
            ("The role of AI in scientific discovery", "science", "medium"),
            ("Implications of the metaverse for society", "metaverse", "medium"),
            ("How 5G will transform mobile applications", "telecommunications", "medium"),
            ("The future of work in a post-pandemic world", "future_work", "medium"),
            ("Blockchain's potential beyond cryptocurrency", "blockchain", "medium"),
            ("The digital divide and technological inequality", "inequality", "hard"),
            ("Edge computing vs cloud computing trade-offs", "infrastructure", "hard"),
            ("The psychology of social media addiction", "psychology", "medium"),
        ]
        
        # Combine all prompt categories
        self.all_prompts = (
            self.programming_prompts + self.ai_ml_prompts + self.ethics_prompts +
            self.creative_prompts + self.career_prompts + self.current_events_prompts
        )
        
    def _get_model_performance_at_time(self, timestamp: datetime) -> Tuple[float, float, float]:
        """Get model accuracy, confidence, and calibration error at specific time"""
        progress = (timestamp - self.start_date).days / self.total_days
        
        # Sigmoid improvement curve (realistic learning)
        sigmoid = 1 / (1 + math.exp(-8 * (progress - 0.5)))
        
        # Accuracy improves over time with some noise
        accuracy = (
            self.initial_accuracy + 
            (self.final_accuracy - self.initial_accuracy) * sigmoid +
            random.gauss(0, 0.03)  # Small random variation
        )
        accuracy = max(0.45, min(0.92, accuracy))  # Bounded
        
        # Confidence starts overconfident, becomes well-calibrated
        confidence_bias = 0.15 * (1 - sigmoid)  # Overconfidence decreases
        base_confidence = accuracy + confidence_bias + random.gauss(0, 0.05)
        confidence = max(0.3, min(0.95, base_confidence))
        
        # Calibration error decreases over time
        cal_error = (
            self.initial_confidence_error + 
            (self.final_confidence_error - self.initial_confidence_error) * sigmoid +
            random.gauss(0, 0.02)
        )
        cal_error = max(0.05, min(0.3, cal_error))
        
        return accuracy, confidence, cal_error
        
    def _get_prompt_mix_at_time(self, timestamp: datetime) -> str:
        """Get the type of prompts more likely at specific time"""
        progress = (timestamp - self.start_date).days / self.total_days
        
        if progress < 0.2:  # Early period: learning basics
            weights = {"programming": 0.4, "ai_ml": 0.3, "career": 0.2, "ethics": 0.1}
        elif progress < 0.5:  # Middle period: expanding domains
            weights = {"programming": 0.3, "ai_ml": 0.3, "ethics": 0.2, "creative": 0.1, "career": 0.1}
        elif progress < 0.8:  # Later period: complex topics
            weights = {"programming": 0.25, "ai_ml": 0.25, "ethics": 0.25, "creative": 0.15, "current_events": 0.1}
        else:  # Recent: diverse usage
            weights = {"programming": 0.2, "ai_ml": 0.2, "ethics": 0.2, "creative": 0.15, "career": 0.15, "current_events": 0.1}
        
        return random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        
    def _generate_realistic_completions(self, prompt: str, category: str, difficulty: str) -> Tuple[str, str]:
        """Generate two realistic completion options"""
        
        base_templates = {
            "programming": [
                "Here's a basic implementation:\n\n```{lang}\n{code}\n```",
                "Here's a comprehensive solution with error handling:\n\n```{lang}\n{code}\n```\n\nThis approach {explanation}"
            ],
            "ai_ml": [
                "{concept} is {basic_explanation}. {applications}",
                "{concept} involves {detailed_explanation}. Key considerations include: 1) {point1}, 2) {point2}, 3) {point3}. {implications}"
            ],
            "ethics": [
                "{stance} because {reason}. However, {consideration}.",
                "This is a complex issue requiring balanced approach: {perspective1} vs {perspective2}. {nuanced_conclusion}"
            ],
            "creative": [
                "{creative_opening} {development} {conclusion}",
                "{detailed_opening} {rich_development} {meaningful_conclusion}"
            ],
            "career": [
                "{advice}: {steps}",
                "{comprehensive_advice}: 1) {step1}, 2) {step2}, 3) {step3}. {additional_context}"
            ]
        }
        
        # Select appropriate templates
        templates = base_templates.get(category, base_templates["programming"])
        
        # Generate different quality levels
        if difficulty == "easy":
            comp_a = templates[0].format(
                lang="python", code="# Simple implementation", explanation="works for basic cases",
                concept="The concept", basic_explanation="straightforward", applications="Common uses include...",
                stance="Yes", reason="it's necessary", consideration="there are challenges",
                creative_opening="Once upon a time", development="things happened", conclusion="The end.",
                advice="Focus on basics", steps="practice daily, read documentation"
            )
            comp_b = templates[1].format(
                lang="python", code="# Robust implementation with error handling", explanation="handles edge cases and provides better user experience",
                concept="The concept", detailed_explanation="multiple interconnected factors", point1="accuracy", point2="fairness", point3="scalability", implications="This affects...",
                perspective1="innovation benefits", perspective2="safety concerns", nuanced_conclusion="The optimal approach balances both",
                detailed_opening="In a world where technology...", rich_development="characters faced complex challenges...", meaningful_conclusion="...leading to profound insights",
                comprehensive_advice="Strategic approach", step1="start small", step2="build projects", step3="get feedback", additional_context="Remember that success takes time"
            )
            return comp_a, comp_b
        else:
            # Return more sophisticated responses for medium/hard
            comp_a = templates[1].format(
                lang="python", code="# Advanced implementation", explanation="uses best practices and optimization",
                concept="The complex concept", detailed_explanation="nuanced interactions", point1="technical depth", point2="ethical considerations", point3="long-term implications", implications="Far-reaching consequences include...",
                perspective1="multiple stakeholder interests", perspective2="competing priorities", nuanced_conclusion="Requires careful balance and ongoing evaluation",
                detailed_opening="The narrative begins with intricate worldbuilding...", rich_development="layered character development and thematic exploration...", meaningful_conclusion="...culminating in thought-provoking resolution",
                comprehensive_advice="Holistic development strategy", step1="understand requirements", step2="design solutions", step3="iterate based on feedback", additional_context="Success requires both technical and soft skills"
            )
            comp_b = templates[0].format(
                lang="python", code="# Efficient solution", explanation="balances performance and readability",
                concept="The concept", basic_explanation="involves key principles", applications="Applications span multiple domains",
                stance="It depends", reason="context matters", consideration="implementation challenges exist",
                creative_opening="The story unfolds", development="with meaningful progression", conclusion="toward satisfying resolution",
                advice="Take systematic approach", steps="assess, plan, execute, review"
            )
            return comp_a, comp_b
    
    def _determine_human_choice(self, prompt: str, comp_a: str, comp_b: str, 
                               model_confidence: float, timestamp: datetime) -> Tuple[int, float]:
        """Determine which completion human would prefer"""
        
        # Factors influencing human choice
        progress = (timestamp - self.start_date).days / self.total_days
        
        # Length preference (humans often prefer more detailed responses)
        length_factor = len(comp_b) / (len(comp_a) + len(comp_b))
        
        # Code formatting preference (prefer code blocks)
        format_factor = 0.6 if "```" in comp_b else 0.4
        
        # Structure preference (prefer numbered lists, explanations)
        structure_factor = 0.6 if any(x in comp_b for x in ["1)", "2)", "•", "-"]) else 0.4
        
        # Complexity preference evolves over time
        if progress < 0.3:  # Early users prefer simpler
            complexity_factor = 0.4 if "comprehensive" in comp_b else 0.6
        else:  # Later users prefer detailed
            complexity_factor = 0.7 if "comprehensive" in comp_b else 0.3
            
        # Combine factors
        prob_choose_b = (length_factor * 0.3 + format_factor * 0.2 + 
                        structure_factor * 0.2 + complexity_factor * 0.3)
        
        # Add some randomness
        prob_choose_b += random.gauss(0, 0.15)
        prob_choose_b = max(0.1, min(0.9, prob_choose_b))
        
        choice = 1 if random.random() < prob_choose_b else 0
        
        # Human confidence based on clarity of preference
        confidence_range = abs(prob_choose_b - 0.5) * 2  # 0 to 1
        human_confidence = 0.5 + confidence_range * 0.4 + random.gauss(0, 0.1)
        human_confidence = max(0.3, min(0.95, human_confidence))
        
        return choice, human_confidence
        
    def generate_votes_data(self) -> List[Dict]:
        """Generate rich votes dataset"""
        votes = []
        
        for i in range(self.total_prompts):
            # Calculate timestamp with realistic usage patterns
            day_offset = int(np.random.weibull(2) * self.total_days)  # Weibull for realistic usage
            day_offset = min(day_offset, self.total_days - 1)
            
            base_time = self.start_date + timedelta(days=day_offset)
            # Add some hour variation (people work different hours)
            hour_offset = np.random.weibull(1.5) * 16 + 6  # 6 AM to 10 PM peak
            hour_offset = int(min(hour_offset, 23))
            minute_offset = random.randint(0, 59)
            
            timestamp = base_time.replace(hour=hour_offset, minute=minute_offset)
            
            # Select prompt based on time period
            category = self._get_prompt_mix_at_time(timestamp)
            
            # Filter prompts by category
            category_prompts = {
                "programming": self.programming_prompts,
                "ai_ml": self.ai_ml_prompts,
                "ethics": self.ethics_prompts,
                "creative": self.creative_prompts,
                "career": self.career_prompts,
                "current_events": self.current_events_prompts
            }
            
            prompt_text, subcategory, difficulty = random.choice(category_prompts[category])
            
            # Generate completions
            comp_a, comp_b = self._generate_realistic_completions(prompt_text, category, difficulty)
            
            # Get model performance at this time
            accuracy, model_confidence, cal_error = self._get_model_performance_at_time(timestamp)
            
            # Determine human choice
            human_choice, human_confidence = self._determine_human_choice(
                prompt_text, comp_a, comp_b, model_confidence, timestamp
            )
            
            vote = {
                "id": f"demo_{i+1:03d}",
                "prompt": prompt_text,
                "completions": [comp_a, comp_b],
                "chosen_index": human_choice,
                "confidence": human_confidence,
                "annotation": f"User preferred response {['A', 'B'][human_choice]} - {self._generate_annotation(category, difficulty, human_choice)}",
                "generation_metadata": {
                    "temperature": round(random.uniform(0.5, 0.9), 2),
                    "max_tokens": random.randint(150, 400),
                    "model": "deepseek-chat",
                    "tokens": {
                        "prompt_tokens": random.randint(30, 120),
                        "completion_tokens": random.randint(80, 350),
                        "total_tokens": random.randint(110, 470)
                    },
                    "cost": round(random.uniform(0.0001, 0.0008), 6),
                    "timestamp": timestamp.isoformat(),
                    "category": category,
                    "subcategory": subcategory,
                    "difficulty": difficulty
                },
                "timestamp": timestamp.isoformat()
            }
            
            votes.append(vote)
            
        # Sort by timestamp
        votes.sort(key=lambda x: x["timestamp"])
        return votes
        
    def _generate_annotation(self, category: str, difficulty: str, choice: int) -> str:
        """Generate realistic human annotation"""
        
        annotations = {
            "programming": [
                "cleaner implementation",
                "better error handling", 
                "more comprehensive solution",
                "clearer code structure",
                "includes best practices"
            ],
            "ai_ml": [
                "more detailed explanation",
                "better examples provided",
                "covers important nuances",
                "more accessible language",
                "practical applications included"
            ],
            "ethics": [
                "more balanced perspective",
                "addresses counterarguments",
                "nuanced analysis",
                "considers multiple stakeholders",
                "practical implications discussed"
            ],
            "creative": [
                "more engaging narrative",
                "stronger character development",
                "more vivid imagery",
                "better pacing",
                "more meaningful conclusion"
            ],
            "career": [
                "more actionable advice",
                "comprehensive strategy",
                "realistic expectations",
                "practical steps provided",
                "addresses common challenges"
            ]
        }
        
        category_annotations = annotations.get(category, annotations["programming"])
        return random.choice(category_annotations)
        
    def generate_predictions_data(self, votes: List[Dict]) -> List[Dict]:
        """Generate predictions dataset matching votes"""
        predictions = []
        
        for vote in votes:
            timestamp = datetime.fromisoformat(vote["timestamp"])
            accuracy, model_confidence, cal_error = self._get_model_performance_at_time(timestamp)
            
            # Model prediction (whether it matches human choice)
            is_correct = random.random() < accuracy
            model_choice = vote["chosen_index"] if is_correct else (1 - vote["chosen_index"])
            
            # Calculate calibrated confidence
            raw_confidence = model_confidence + random.gauss(0, 0.05)
            raw_confidence = max(0.2, min(0.95, raw_confidence))
            
            # Temperature scaling for calibration
            temperature = 1.0 + cal_error * 2  # Higher temp for poor calibration
            calibrated_confidence = raw_confidence ** (1/temperature)
            calibrated_confidence = max(0.1, min(0.95, calibrated_confidence))
            
            pred = {
                "prompt_id": vote["id"],
                "pair_id": f"{vote['id']}_pair",
                "prompt": vote["prompt"],
                "completion_a": vote["completions"][0][:100] + "...",  # Truncated for storage
                "completion_b": vote["completions"][1][:100] + "...",
                "choice": ["A", "B"][model_choice],
                "confidence": round(calibrated_confidence, 3),
                "is_model_vote": True,
                "timestamp": vote["timestamp"],
                "raw_prediction": {
                    "preferred_completion": ["A", "B"][model_choice],
                    "confidence": round(calibrated_confidence, 3),
                    "raw_confidence": round(raw_confidence, 3),
                    "calibrated_probabilities": [
                        round(1 - calibrated_confidence, 3),
                        round(calibrated_confidence, 3)
                    ] if model_choice == 1 else [
                        round(calibrated_confidence, 3),
                        round(1 - calibrated_confidence, 3)
                    ],
                    "raw_probabilities": [
                        round(1 - raw_confidence, 3),
                        round(raw_confidence, 3)
                    ] if model_choice == 1 else [
                        round(raw_confidence, 3),
                        round(1 - raw_confidence, 3)
                    ],
                    "calibration_method": "temperature_scaling"
                },
                "is_confident": calibrated_confidence > 0.7
            }
            
            predictions.append(pred)
            
        return predictions
        
    def generate_reflection_data(self, votes: List[Dict], predictions: List[Dict]) -> List[Dict]:
        """Generate reflection dataset analyzing human vs model choices"""
        reflections = []
        
        for vote, pred in zip(votes, predictions):
            human_choice = vote["chosen_index"]
            model_choice = 0 if pred["choice"] == "A" else 1
            is_correct = human_choice == model_choice
            
            # Error type analysis
            error_type = None
            if not is_correct:
                if pred["confidence"] > 0.7:
                    error_type = "overconfident_error"
                elif pred["confidence"] < 0.4:
                    error_type = "underconfident_error"
                else:
                    error_type = "false_prediction"
                    
            # Quality metrics based on category
            category = vote["generation_metadata"]["category"]
            quality_metrics = self._generate_quality_metrics(category, is_correct)
            
            reflection = {
                "timestamp": vote["timestamp"],
                "vote_timestamp": vote["timestamp"],
                "prompt": vote["prompt"],
                "human_choice": human_choice,
                "model_prediction": model_choice,
                "is_correct": is_correct,
                "human_confidence": vote["confidence"],
                "model_confidence": pred["confidence"],
                "confidence_gap": round(vote["confidence"] - pred["confidence"], 3),
                "error_type": error_type,
                "model_probabilities": pred["raw_prediction"]["calibrated_probabilities"],
                "model_logits": [
                    round(math.log(p / (1 - p + 1e-10)), 2) for p in pred["raw_prediction"]["calibrated_probabilities"]
                ],
                "original_vote_metadata": {
                    "prompt_id": vote["id"],
                    "preference": f"Completion {['A', 'B'][human_choice]}",
                    "feedback": vote["annotation"],
                    "quality_metrics": quality_metrics
                }
            }
            
            reflections.append(reflection)
            
        return reflections
        
    def _generate_quality_metrics(self, category: str, is_correct: bool) -> Dict[str, float]:
        """Generate realistic quality metrics"""
        base_score = 0.8 if is_correct else 0.6
        variation = 0.15
        
        metrics = {
            "programming": ["technical_accuracy", "code_quality", "clarity"],
            "ai_ml": ["clarity", "technical_accuracy", "accessibility"],
            "ethics": ["balance", "depth", "practical_value"],
            "creative": ["creativity", "emotional_impact", "narrative_structure"],
            "career": ["practical_value", "comprehensiveness", "specificity"]
        }
        
        metric_names = metrics.get(category, metrics["programming"])
        
        return {
            metric: round(base_score + random.gauss(0, variation), 2)
            for metric in metric_names
        }
        
    def generate_calibration_history(self) -> Dict[str, Any]:
        """Generate comprehensive calibration history"""
        
        # Generate calibration events over time
        calibration_events = []
        event_dates = []
        
        # Generate calibration sessions every 2-3 weeks
        current_date = self.start_date + timedelta(days=14)
        while current_date < self.end_date:
            event_dates.append(current_date)
            current_date += timedelta(days=random.randint(14, 21))
            
        for i, event_date in enumerate(event_dates):
            progress = (event_date - self.start_date).days / self.total_days
            
            # Model improves over time
            pre_ece = self.initial_confidence_error * (1 - progress * 0.7)
            post_ece = pre_ece * 0.6  # Calibration reduces error
            
            sample_count = random.randint(80, 150)
            accuracy = self.initial_accuracy + (self.final_accuracy - self.initial_accuracy) * progress
            
            event = {
                "timestamp": event_date.isoformat(),
                "sample_count": sample_count,
                "accuracy": round(accuracy, 3),
                "calibration_error": round(post_ece, 4),
                "avg_confidence_before": round(accuracy + pre_ece, 3),
                "avg_confidence_after": round(accuracy + post_ece * 0.3, 3),
                "temperature": round(1.0 + pre_ece, 3),
                "notes": f"Calibration session {i+1} - {'Good improvement' if i > 2 else 'Initial calibration'}"
            }
            calibration_events.append(event)
            
        # Calculate final metrics
        latest_event = calibration_events[-1]
        
        calibration_data = {
            "timestamp": datetime.now().isoformat(),
            "method": "temperature_scaling",
            "parameters": {
                "temperature": latest_event["temperature"],
                "validation_samples": latest_event["sample_count"]
            },
            "metrics": {
                "pre_calibration": {
                    "ece": round(self.initial_confidence_error, 4),
                    "log_loss": round(0.65, 3),
                    "brier_score": round(0.25, 3)
                },
                "post_calibration": {
                    "ece": round(self.final_confidence_error, 4),
                    "log_loss": round(0.35, 3),
                    "brier_score": round(0.18, 3)
                },
                "improvement": {
                    "ece": round(self.initial_confidence_error - self.final_confidence_error, 4),
                    "log_loss": round(0.30, 3),
                    "brier_score": round(0.07, 3)
                }
            },
            "history": calibration_events
        }
        
        return calibration_data
        
    def generate_model_checkpoints(self) -> List[Dict]:
        """Generate model checkpoint metadata"""
        checkpoints = []
        
        # Generate checkpoints at major training milestones
        checkpoint_dates = [
            self.start_date + timedelta(days=30),   # v1 - initial
            self.start_date + timedelta(days=90),   # v2 - first major update  
            self.start_date + timedelta(days=150),  # v3 - significant improvement
            self.start_date + timedelta(days=180),  # v4 - latest stable
        ]
        
        for i, date in enumerate(checkpoint_dates):
            progress = (date - self.start_date).days / self.total_days
            accuracy = self.initial_accuracy + (self.final_accuracy - self.initial_accuracy) * progress
            
            checkpoint = {
                "version": f"v{i+1}",
                "timestamp": date.isoformat(),
                "accuracy": round(accuracy, 3),
                "calibration_error": round(self.initial_confidence_error * (1 - progress * 0.8), 4),
                "training_samples": random.randint(500, 2000),
                "notes": [
                    "Initial model training with basic RLHF dataset",
                    "Enhanced training with more diverse prompts",
                    "Major architecture improvements and calibration",
                    "Latest stable release with optimal performance"
                ][i],
                "model_size": random.choice(["7B", "13B", "20B"]),
                "training_time_hours": random.randint(12, 48),
                "confidence_avg": round(accuracy + random.gauss(0, 0.05), 3),
                "loss": round(random.uniform(0.3, 0.8), 3)
            }
            
            checkpoints.append(checkpoint)
            
        return checkpoints
        
    def generate_all_data(self):
        """Generate all demo data files"""
        print("🚀 Generating rich demo data...")
        print(f"   📅 Time period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"   📊 Total prompts: {self.total_prompts}")
        print(f"   🎯 Model evolution: {self.initial_accuracy:.1%} → {self.final_accuracy:.1%} accuracy")
        
        # Generate main datasets
        print("\n📝 Generating votes data...")
        votes = self.generate_votes_data()
        
        print("🤖 Generating predictions data...")
        predictions = self.generate_predictions_data(votes)
        
        print("🔍 Generating reflection data...")
        reflections = self.generate_reflection_data(votes, predictions)
        
        print("⚖️ Generating calibration history...")
        calibration = self.generate_calibration_history()
        
        print("🏗️ Generating model checkpoints...")
        checkpoints = self.generate_model_checkpoints()
        
        # Write data files
        print("\n💾 Writing data files...")
        
        # Main demo files
        with open(DATA_DIR / "demo_votes.jsonl", 'w') as f:
            for vote in votes:
                f.write(json.dumps(vote) + '\n')
        print(f"   ✓ demo_votes.jsonl ({len(votes)} entries)")
        
        with open(DATA_DIR / "demo_predictions.jsonl", 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
        print(f"   ✓ demo_predictions.jsonl ({len(predictions)} entries)")
        
        with open(DATA_DIR / "demo_reflection_data.jsonl", 'w') as f:
            for refl in reflections:
                f.write(json.dumps(refl) + '\n')
        print(f"   ✓ demo_reflection_data.jsonl ({len(reflections)} entries)")
        
        # Calibration data
        calib_dir = DATA_DIR.parent / "models"
        calib_dir.mkdir(exist_ok=True)
        with open(calib_dir / "demo_calibration_log.json", 'w') as f:
            json.dump(calibration, f, indent=2)
        print(f"   ✓ models/demo_calibration_log.json")
        
        # Model checkpoints
        checkpoint_dir = calib_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        for i, checkpoint in enumerate(checkpoints):
            with open(checkpoint_dir / f"demo_checkpoint_v{i+1}_metadata.json", 'w') as f:
                json.dump(checkpoint, f, indent=2)
        print(f"   ✓ models/checkpoints/ ({len(checkpoints)} checkpoints)")
        
        # Generate some individual vote logs for vote_logs/ directory
        vote_logs_dir = DATA_DIR / "vote_logs"
        vote_logs_dir.mkdir(exist_ok=True)
        
        # Create individual vote files for recent votes
        recent_votes = votes[-20:]  # Last 20 votes
        for vote in recent_votes:
            vote_filename = f"demo_vote_{vote['timestamp'][:10]}_{vote['id']}.json"
            with open(vote_logs_dir / vote_filename, 'w') as f:
                json.dump(vote, f, indent=2)
        print(f"   ✓ data/vote_logs/ ({len(recent_votes)} individual vote files)")
        
        # Statistics summary
        print(f"\n📈 Dataset Statistics:")
        print(f"   🎯 Model accuracy improvement: {self.initial_accuracy:.1%} → {self.final_accuracy:.1%}")
        print(f"   🎚️ Calibration improvement: {self.initial_confidence_error:.3f} → {self.final_confidence_error:.3f} ECE")
        print(f"   📅 Time span: {(self.end_date - self.start_date).days} days")
        
        # Category breakdown
        categories = {}
        for vote in votes:
            cat = vote["generation_metadata"]["category"]
            categories[cat] = categories.get(cat, 0) + 1
            
        print(f"   📚 Content categories:")
        for cat, count in categories.items():
            print(f"     • {cat}: {count} prompts ({count/len(votes):.1%})")
            
        print(f"\n✅ Rich demo data generation complete!")
        print(f"   💎 {len(votes)} diverse prompts spanning {len(categories)} domains")
        print(f"   🏆 Perfect for portfolio showcases and feature exploration")
        print(f"\n   🎮 Run demo mode to see this rich data in action!")

def main():
    """Main entry point"""
    generator = RichDemoDataGenerator()
    generator.generate_all_data()

if __name__ == "__main__":
    main() 