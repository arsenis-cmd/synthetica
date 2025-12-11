#!/usr/bin/env python3
"""
Example script to generate synthetic customer support conversations.

This script demonstrates how to use Synthetica to generate, score, and export
synthetic customer support conversations.
"""
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to import synthetica
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetica.generators.conversation import ConversationGenerator
from synthetica.quality.scorer import QualityScorer
from synthetica.schemas.conversation import ConversationConfig
from synthetica.output.formatters import ConversationFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Generate sample conversations and export them."""
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        logger.info("Please set your Anthropic API key:")
        logger.info("  export ANTHROPIC_API_KEY='your-api-key-here'")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Synthetica - Customer Support Conversation Generator")
    logger.info("=" * 60)

    # Initialize generator and scorer
    logger.info("Initializing generator and scorer...")
    generator = ConversationGenerator(api_key=api_key)
    scorer = QualityScorer(api_key=api_key)

    # Define diverse configurations for variety
    configs = [
        ConversationConfig(
            industry="technology",
            topics=["software bug", "account access", "feature request"],
            tone="professional",
            message_count=8,
            company_name="TechCorp"
        ),
        ConversationConfig(
            industry="e-commerce",
            topics=["order tracking", "return request", "product inquiry"],
            tone="casual",
            message_count=6,
            company_name="ShopEasy"
        ),
        ConversationConfig(
            industry="banking",
            topics=["transaction dispute", "card activation", "loan inquiry"],
            tone="formal",
            message_count=10,
            company_name="SecureBank"
        ),
        ConversationConfig(
            industry="healthcare",
            topics=["appointment scheduling", "prescription refill", "insurance question"],
            tone="empathetic",
            message_count=8,
            company_name="HealthFirst"
        ),
        ConversationConfig(
            industry="telecommunications",
            topics=["service outage", "plan upgrade", "billing question"],
            tone="professional",
            message_count=8,
            company_name="ConnectTel"
        ),
    ]

    # Generate conversations
    logger.info("\nGenerating 10 conversations across different industries...")
    all_conversations = []

    for i in range(10):
        # Cycle through configurations
        config = configs[i % len(configs)]

        try:
            logger.info(f"\n[{i+1}/10] Generating {config.industry} conversation...")
            conversation = generator.generate(config)

            logger.info(f"  ✓ Generated conversation {conversation.id}")
            logger.info(f"  - Industry: {config.industry}")
            logger.info(f"  - Topic: {conversation.metadata.topic}")
            logger.info(f"  - Messages: {len(conversation.messages)}")

            # Score the conversation (using heuristics for speed)
            logger.info("  - Scoring quality...")
            quality = scorer.score(conversation, use_ai=False)
            conversation.quality_score = quality.overall

            logger.info(f"  - Quality Score: {quality.overall:.1f}/100")
            logger.info(f"    • Coherence: {quality.coherence:.1f}")
            logger.info(f"    • Diversity: {quality.diversity:.1f}")
            logger.info(f"    • Naturalness: {quality.naturalness:.1f}")

            all_conversations.append(conversation)

        except Exception as e:
            logger.error(f"  ✗ Failed to generate conversation {i+1}: {e}")
            continue

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"Successfully generated {len(all_conversations)}/10 conversations")

    if all_conversations:
        avg_quality = sum(c.quality_score for c in all_conversations if c.quality_score) / len(all_conversations)
        logger.info(f"Average quality score: {avg_quality:.1f}/100")

        # Export to all formats
        logger.info("\nExporting conversations...")
        output_dir = Path(__file__).parent.parent / "output"

        ConversationFormatter.save_all_formats(
            all_conversations,
            str(output_dir),
            name="sample_conversations"
        )

        logger.info(f"\n✓ Exported to:")
        logger.info(f"  - JSON: {output_dir}/sample_conversations.json")
        logger.info(f"  - JSONL: {output_dir}/sample_conversations.jsonl")
        logger.info(f"  - CSV: {output_dir}/sample_conversations.csv")
        logger.info(f"  - CSV (detailed): {output_dir}/sample_conversations_detailed.csv")

        # Display sample conversation
        logger.info("\n" + "=" * 60)
        logger.info("Sample Conversation Preview:")
        logger.info("=" * 60)

        sample = all_conversations[0]
        logger.info(f"\nID: {sample.id}")
        logger.info(f"Industry: {sample.metadata.industry}")
        logger.info(f"Topic: {sample.metadata.topic}")
        logger.info(f"Quality: {sample.quality_score:.1f}/100\n")

        for msg in sample.messages:
            role_label = "👤 CUSTOMER" if msg.role == "customer" else "👨‍💼 AGENT"
            logger.info(f"{role_label}: {msg.content}\n")

    logger.info("=" * 60)
    logger.info("Generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
