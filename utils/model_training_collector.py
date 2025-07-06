#!/usr/bin/env python3
"""
Model Training Data Collector

This script demonstrates how to collect and export training data
for fine-tuning RLHF models to improve performance and accuracy.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

from utils.database import get_database
from datetime import datetime

def collect_sample_interactions():
    """Collect some sample interactions for demonstration."""
    db = get_database()
    
    print("📊 Model Training Data Collector - Building Performance Dataset")
    print("=" * 70)
    
    # Sample conversations to demonstrate the system
    sample_conversations = [
        {
            "user_message": "Tell me about the model's capabilities",
            "model_response": "I am an AI assistant designed to be helpful, accurate, and responsive to user needs. I learn from interactions to improve my performance and provide better assistance over time.",
            "model_mode": "standard",
            "response_type": "informative",
            "user_feedback": "positive"
        },
        {
            "user_message": "How do you learn from feedback?",
            "model_response": "I learn from feedback by analyzing user preferences, identifying patterns in successful interactions, and adjusting my responses to better align with user expectations and needs.",
            "model_mode": "standard", 
            "response_type": "analytical",
            "user_feedback": "positive"
        },
        {
            "user_message": "What makes a good response?",
            "model_response": "A good response is accurate, helpful, clear, and relevant to the user's question. It should provide useful information while being concise and easy to understand.",
            "model_mode": "standard",
            "response_type": "educational",
            "user_feedback": "positive"
        }
    ]
    
    print("📝 Saving sample conversations for training...")
    for i, conv in enumerate(sample_conversations, 1):
        success = db.save_conversation_for_training(
            user_message=conv["user_message"],
            model_response=conv["model_response"],
            model_mode=conv["model_mode"],
            response_type=conv["response_type"],
            user_feedback=conv["user_feedback"]
        )
        
        if success:
            print(f"✅ Saved conversation {i}: {conv['user_message'][:50]}...")
        else:
            print(f"❌ Failed to save conversation {i}")
    
    # Save some reflections
    print("\n💭 Saving model analysis reflections...")
    reflections = [
        "User feedback patterns show preference for clear, concise responses with specific examples.",
        "Model performance improves when responses are tailored to the user's apparent expertise level.",
        "Successful interactions often involve asking clarifying questions when the user's intent is unclear."
    ]
    
    for i, reflection in enumerate(reflections, 1):
        success = db.save_model_reflection(reflection, "performance")
        if success:
            print(f"✅ Saved reflection {i}")
        else:
            print(f"❌ Failed to save reflection {i}")

def export_training_dataset():
    """Export the complete training dataset."""
    db = get_database()
    
    print("\n📦 Exporting complete training dataset...")
    try:
        dataset_path = db.export_model_training_dataset()
        print(f"✅ Successfully exported training dataset to: {dataset_path}")
        
        # Show some statistics
        metrics = db.get_model_performance_metrics()
        print(f"\n📊 Dataset Statistics:")
        print(f"   📈 Total interactions: {metrics['total_interactions']}")
        print(f"   🎯 Model accuracy: {metrics['model_accuracy']:.2%}")
        print(f"   📊 Learning progression: {metrics['learning_progression']:.2%}")
        print(f"   🔧 Learning progress: {metrics['learning_progress']:.2%}")
        print(f"   ✅ Performance indicators: {len(metrics['performance_indicators'])}")
        
        return dataset_path
        
    except Exception as e:
        print(f"❌ Failed to export dataset: {e}")
        return None

def show_learning_stages():
    """Show model learning progression through different stages."""
    db = get_database()
    
    print("\n📈 Model Learning Stages:")
    print("=" * 50)
    
    stages = db.get_model_learning_stages()
    if not stages:
        print("No learning data available yet. Start interacting to build training history!")
        return
    
    for stage in stages:
        print(f"\n🔄 {stage['stage']}")
        print(f"   📅 Period: {stage['period']}")
        print(f"   📊 Annotations: {stage['annotations']}")
        print(f"   🎯 Accuracy: {stage['accuracy']:.2%}")
        print(f"   💡 Confidence Level: {stage['confidence_level']:.2%}")
        print(f"   📈 Learning Rate: {stage['learning_rate']:.2%}")
        print(f"   💭 Analysis: {stage['analysis']}")

def main():
    """Main function to demonstrate the training data collection system."""
    print("🔧 Welcome to the Model Training Data Collection System!")
    print("This system captures interaction data for model improvement and fine-tuning.")
    print()
    
    # Collect sample data
    collect_sample_interactions()
    
    # Show learning stages
    show_learning_stages()
    
    # Export dataset
    dataset_path = export_training_dataset()
    
    print("\n" + "=" * 70)
    print("📊 Training Data Collection Complete!")
    print()
    print("🎯 What this system captures for fine-tuning:")
    print("   • User interactions and model responses")
    print("   • Response types and quality metrics")
    print("   • User feedback and preferences")
    print("   • Model learning progression")
    print("   • Performance analytics")
    print("   • Quality assessments and improvements")
    print()
    print("🚀 How to use this for fine-tuning:")
    print("   1. Interact with the model through the interface")
    print("   2. Provide feedback on model responses")
    print("   3. Export the dataset periodically")
    print("   4. Use the dataset to fine-tune the language model")
    print("   5. Deploy the improved model version")
    print()
    if dataset_path:
        print(f"📁 Your training dataset is ready at: {dataset_path}")
    print("📈 Every interaction helps improve model performance!")

if __name__ == "__main__":
    main()
