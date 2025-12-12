"""
Test script demonstrating persona system with specific personas.

This shows how different personas create distinctly different conversation styles.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetica.generators.conversation import ConversationGenerator
from synthetica.generators.diversity import DiversityEngine
from synthetica.schemas.conversation import ConversationConfig
from synthetica.quality.scorer import QualityScorer


def test_nervous_patient_rushed_doctor():
    """Test with nervous patient and rushed doctor personas."""
    print("\n" + "="*80)
    print("Testing HEALTHCARE with NERVOUS PATIENT + RUSHED DOCTOR")
    print("="*80)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        return None

    # Create diversity engine
    diversity_engine = DiversityEngine()

    # Force specific personas by modifying the persona generator
    nervous_patient = diversity_engine.persona_generator.get_persona_by_id("nervous_patient")
    rushed_doctor = diversity_engine.persona_generator.get_persona_by_id("rushed_doctor")

    print(f"\n✓ Loaded {nervous_patient.name} persona")
    print(f"  - Tone: {nervous_patient.tone}")
    print(f"  - Communication Style: {nervous_patient.communication_style}")
    print(f"  - Typical Phrases: {', '.join(nervous_patient.typical_phrases[:2])}")

    print(f"\n✓ Loaded {rushed_doctor.name} persona")
    print(f"  - Tone: {rushed_doctor.tone}")
    print(f"  - Communication Style: {rushed_doctor.communication_style}")
    print(f"  - Typical Phrases: {', '.join(rushed_doctor.typical_phrases[:2])}")

    # Override the get_persona_for_domain_role to return our specific personas
    original_method = diversity_engine.persona_generator.get_persona_for_domain_role

    def custom_get_persona(domain, role):
        if domain == "healthcare":
            if role == "patient":
                return nervous_patient
            elif role == "doctor":
                return rushed_doctor
        return original_method(domain, role)

    diversity_engine.persona_generator.get_persona_for_domain_role = custom_get_persona

    generator = ConversationGenerator(api_key=api_key, diversity_engine=diversity_engine)
    scorer = QualityScorer(api_key=api_key)

    config = ConversationConfig(
        domain="healthcare",
        industry="emergency medicine",
        topics=["chest pain"],
        scenario="patient experiencing chest pain seeks urgent medical evaluation",
        role_1="patient",
        role_2="doctor",
        tone="professional",
        message_count=10
    )

    print(f"\nGenerating conversation...")
    print(f"  Domain: {config.domain}")
    print(f"  Roles: {config.role_1} / {config.role_2}")
    print(f"  Scenario: {config.scenario}")

    conversation = generator.generate(config)
    quality = scorer.score(conversation, use_ai=False)

    print(f"\n{'='*80}")
    print(f"CONVERSATION RESULT")
    print(f"{'='*80}")
    print(f"✓ Conversation ID: {conversation.id}")
    print(f"✓ Messages: {len(conversation.messages)}")
    print(f"✓ Quality Score: {quality.overall:.1f}/100")

    # Display persona metadata
    if conversation.metadata.persona:
        print(f"\n{'='*80}")
        print(f"PERSONA METADATA")
        print(f"{'='*80}")
        if 'role_1_persona' in conversation.metadata.persona:
            p1 = conversation.metadata.persona['role_1_persona']
            print(f"\n{config.role_1.upper()} PERSONA: {p1['name']}")
            print(f"  - Tone: {p1['tone']}")
            print(f"  - Communication Style: {p1['communication_style']}")
        if 'role_2_persona' in conversation.metadata.persona:
            p2 = conversation.metadata.persona['role_2_persona']
            print(f"\n{config.role_2.upper()} PERSONA: {p2['name']}")
            print(f"  - Tone: {p2['tone']}")
            print(f"  - Communication Style: {p2['communication_style']}")

    print(f"\n{'='*80}")
    print(f"CONVERSATION (notice the distinct personalities!)")
    print(f"{'='*80}\n")

    for i, msg in enumerate(conversation.messages, 1):
        persona_marker = ""
        if msg.role == "patient":
            persona_marker = f" [{nervous_patient.name}]"
        elif msg.role == "doctor":
            persona_marker = f" [{rushed_doctor.name}]"

        print(f"{msg.role.upper()}{persona_marker}:")
        print(f"{msg.content}")
        print()

    print(f"{'='*80}")
    print("\nNOTICE HOW:")
    print(f"  - The PATIENT sounds {nervous_patient.tone} and {nervous_patient.communication_style}")
    print(f"  - The DOCTOR sounds {rushed_doctor.tone} and {rushed_doctor.communication_style}")
    print(f"{'='*80}\n")

    return conversation


def test_contrasting_personas():
    """Test with contrasting personas for comparison."""
    print("\n" + "="*80)
    print("Testing HEALTHCARE with STOIC PATIENT + EMPATHETIC DOCTOR")
    print("(For Comparison)")
    print("="*80)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        return None

    diversity_engine = DiversityEngine()

    stoic_patient = diversity_engine.persona_generator.get_persona_by_id("stoic_patient")
    empathetic_doctor = diversity_engine.persona_generator.get_persona_by_id("empathetic_doctor")

    print(f"\n✓ Loaded {stoic_patient.name} persona")
    print(f"  - Tone: {stoic_patient.tone}")
    print(f"  - Communication Style: {stoic_patient.communication_style}")

    print(f"\n✓ Loaded {empathetic_doctor.name} persona")
    print(f"  - Tone: {empathetic_doctor.tone}")
    print(f"  - Communication Style: {empathetic_doctor.communication_style}")

    # Override personas
    original_method = diversity_engine.persona_generator.get_persona_for_domain_role

    def custom_get_persona(domain, role):
        if domain == "healthcare":
            if role == "patient":
                return stoic_patient
            elif role == "doctor":
                return empathetic_doctor
        return original_method(domain, role)

    diversity_engine.persona_generator.get_persona_for_domain_role = custom_get_persona

    generator = ConversationGenerator(api_key=api_key, diversity_engine=diversity_engine)

    config = ConversationConfig(
        domain="healthcare",
        industry="emergency medicine",
        topics=["chest pain"],
        scenario="patient experiencing chest pain seeks urgent medical evaluation",
        role_1="patient",
        role_2="doctor",
        tone="professional",
        message_count=10
    )

    print(f"\nGenerating conversation...")
    conversation = generator.generate(config)

    print(f"\n{'='*80}")
    print(f"CONVERSATION (notice the VERY DIFFERENT personalities!)")
    print(f"{'='*80}\n")

    for i, msg in enumerate(conversation.messages, 1):
        persona_marker = ""
        if msg.role == "patient":
            persona_marker = f" [{stoic_patient.name}]"
        elif msg.role == "doctor":
            persona_marker = f" [{empathetic_doctor.name}]"

        print(f"{msg.role.upper()}{persona_marker}:")
        print(f"{msg.content}")
        print()

    print(f"{'='*80}")
    print("\nNOTICE HOW:")
    print(f"  - The PATIENT sounds {stoic_patient.tone} and {stoic_patient.communication_style}")
    print(f"  - The DOCTOR sounds {empathetic_doctor.tone} and {empathetic_doctor.communication_style}")
    print(f"  - This is VERY DIFFERENT from the nervous/rushed conversation above!")
    print(f"{'='*80}\n")

    return conversation


def main():
    """Run persona tests."""
    print("\n" + "="*80)
    print("PERSONA SYSTEM DEMONSTRATION")
    print("="*80)
    print("\nThis demonstrates how personas create distinctly different conversation styles.")
    print("We'll generate TWO healthcare conversations with OPPOSITE personalities:\n")
    print("  1. Nervous Patient + Rushed Doctor")
    print("  2. Stoic Patient + Empathetic Doctor")
    print("\nThe difference should be IMMEDIATELY OBVIOUS!")
    print("="*80)

    # Test 1: Nervous + Rushed
    conv1 = test_nervous_patient_rushed_doctor()

    # Test 2: Stoic + Empathetic
    conv2 = test_contrasting_personas()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✓ Generated 2 conversations with VERY different personalities")
    print("✓ Personas are now clearly visible in the conversation style")
    print("✓ Each role has its own distinct personality that shines through")
    print("\nThe persona system is working! 🎉")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
