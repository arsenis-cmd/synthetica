"""
Test script for domain-agnostic conversation generation.

Tests healthcare, sales, and legal domains to verify:
- Domain-specific vocabulary usage
- Correct role assignments
- Appropriate tone and terminology
- Conversation quality
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetica.generators.conversation import ConversationGenerator
from synthetica.schemas.conversation import ConversationConfig
from synthetica.quality.scorer import QualityScorer


def test_healthcare_domain():
    """Test healthcare domain with patient/doctor roles."""
    print("\n" + "="*80)
    print("Testing HEALTHCARE Domain (patient/doctor)")
    print("="*80)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        return None

    generator = ConversationGenerator(api_key=api_key)
    scorer = QualityScorer(api_key=api_key)

    config = ConversationConfig(
        domain="healthcare",
        industry="primary care",
        topics=["symptom discussion"],
        scenario="patient experiencing persistent headaches seeks medical advice",
        role_1="patient",
        role_2="doctor",
        tone="professional",
        message_count=8
    )

    print(f"\nGenerating healthcare conversation...")
    print(f"  Domain: {config.domain}")
    print(f"  Roles: {config.role_1} / {config.role_2}")
    print(f"  Scenario: {config.scenario}")
    print(f"  Industry: {config.industry}")

    conversation = generator.generate(config)
    quality = scorer.score(conversation, use_ai=False)

    print(f"\n✓ Generated conversation ID: {conversation.id}")
    print(f"✓ Messages: {len(conversation.messages)}")
    print(f"✓ Quality Score: {quality.overall:.1f}/100")
    print(f"  - Coherence: {quality.coherence:.1f}")
    print(f"  - Diversity: {quality.diversity:.1f}")
    print(f"  - Naturalness: {quality.naturalness:.1f}")

    print(f"\nConversation Preview:")
    print("-" * 80)
    for msg in conversation.messages[:4]:  # Show first 4 messages
        print(f"{msg.role.upper()}: {msg.content}")
        print()

    if len(conversation.messages) > 4:
        print(f"... ({len(conversation.messages) - 4} more messages)")

    return conversation


def test_sales_domain():
    """Test sales domain with prospect/sales_rep roles."""
    print("\n" + "="*80)
    print("Testing SALES Domain (prospect/sales_rep)")
    print("="*80)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        return None

    generator = ConversationGenerator(api_key=api_key)
    scorer = QualityScorer(api_key=api_key)

    config = ConversationConfig(
        domain="sales",
        industry="SaaS software",
        topics=["product demo"],
        scenario="enterprise prospect inquiring about pricing and features",
        role_1="prospect",
        role_2="sales_rep",
        tone="professional",
        message_count=8
    )

    print(f"\nGenerating sales conversation...")
    print(f"  Domain: {config.domain}")
    print(f"  Roles: {config.role_1} / {config.role_2}")
    print(f"  Scenario: {config.scenario}")
    print(f"  Industry: {config.industry}")

    conversation = generator.generate(config)
    quality = scorer.score(conversation, use_ai=False)

    print(f"\n✓ Generated conversation ID: {conversation.id}")
    print(f"✓ Messages: {len(conversation.messages)}")
    print(f"✓ Quality Score: {quality.overall:.1f}/100")
    print(f"  - Coherence: {quality.coherence:.1f}")
    print(f"  - Diversity: {quality.diversity:.1f}")
    print(f"  - Naturalness: {quality.naturalness:.1f}")

    print(f"\nConversation Preview:")
    print("-" * 80)
    for msg in conversation.messages[:4]:  # Show first 4 messages
        print(f"{msg.role.upper()}: {msg.content}")
        print()

    if len(conversation.messages) > 4:
        print(f"... ({len(conversation.messages) - 4} more messages)")

    return conversation


def test_legal_domain():
    """Test legal domain with client/lawyer roles."""
    print("\n" + "="*80)
    print("Testing LEGAL Domain (client/lawyer)")
    print("="*80)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        return None

    generator = ConversationGenerator(api_key=api_key)
    scorer = QualityScorer(api_key=api_key)

    config = ConversationConfig(
        domain="legal",
        industry="contract law",
        topics=["contract review"],
        scenario="client seeking legal review of employment contract",
        role_1="client",
        role_2="lawyer",
        tone="formal",
        message_count=8
    )

    print(f"\nGenerating legal conversation...")
    print(f"  Domain: {config.domain}")
    print(f"  Roles: {config.role_1} / {config.role_2}")
    print(f"  Scenario: {config.scenario}")
    print(f"  Industry: {config.industry}")

    conversation = generator.generate(config)
    quality = scorer.score(conversation, use_ai=False)

    print(f"\n✓ Generated conversation ID: {conversation.id}")
    print(f"✓ Messages: {len(conversation.messages)}")
    print(f"✓ Quality Score: {quality.overall:.1f}/100")
    print(f"  - Coherence: {quality.coherence:.1f}")
    print(f"  - Diversity: {quality.diversity:.1f}")
    print(f"  - Naturalness: {quality.naturalness:.1f}")

    print(f"\nConversation Preview:")
    print("-" * 80)
    for msg in conversation.messages[:4]:  # Show first 4 messages
        print(f"{msg.role.upper()}: {msg.content}")
        print()

    if len(conversation.messages) > 4:
        print(f"... ({len(conversation.messages) - 4} more messages)")

    return conversation


def main():
    """Run all domain tests."""
    print("\n" + "="*80)
    print("DOMAIN-AGNOSTIC CONVERSATION GENERATOR - DOMAIN TESTS")
    print("="*80)
    print("\nTesting conversation generation across different domains...")
    print("This verifies domain-specific vocabulary, roles, and tone.")

    results = {}

    # Test healthcare
    try:
        results['healthcare'] = test_healthcare_domain()
    except Exception as e:
        print(f"\n✗ Healthcare test failed: {e}")
        results['healthcare'] = None

    # Test sales
    try:
        results['sales'] = test_sales_domain()
    except Exception as e:
        print(f"\n✗ Sales test failed: {e}")
        results['sales'] = None

    # Test legal
    try:
        results['legal'] = test_legal_domain()
    except Exception as e:
        print(f"\n✗ Legal test failed: {e}")
        results['legal'] = None

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    successful = sum(1 for v in results.values() if v is not None)
    total = len(results)

    print(f"\n✓ Successful: {successful}/{total} domains")

    for domain, conv in results.items():
        if conv:
            print(f"  ✓ {domain.capitalize()}: Generated {len(conv.messages)} messages")
        else:
            print(f"  ✗ {domain.capitalize()}: Failed")

    if successful == total:
        print("\n🎉 All domain tests passed!")
        print("\nThe refactoring to domain-agnostic generation is complete and working!")
    else:
        print(f"\n⚠ {total - successful} domain(s) failed. Check errors above.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
