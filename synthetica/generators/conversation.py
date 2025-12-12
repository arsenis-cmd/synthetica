"""
Domain-agnostic conversation generator using Claude API.
"""
import json
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from anthropic import Anthropic

from synthetica.schemas.conversation import (
    Conversation,
    ConversationConfig,
    ConversationMetadata,
    Message,
)
from synthetica.generators.diversity import DiversityEngine, Persona
from synthetica.generators.domain_vocabulary import DomainVocabulary

logger = logging.getLogger(__name__)


class ConversationGenerator:
    """Generates synthetic customer support conversations using Claude API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        diversity_engine: Optional[DiversityEngine] = None
    ):
        """
        Initialize the conversation generator.

        Args:
            api_key: Anthropic API key
            model: Claude model to use for generation
            diversity_engine: Optional diversity engine for persona and style variation
        """
        self.client = Anthropic(
            api_key=api_key,
            timeout=60.0,  # Increase timeout to 60 seconds for Railway
            max_retries=3
        )
        self.model = model
        self.diversity_engine = diversity_engine

    def generate(self, config: ConversationConfig) -> Conversation:
        """
        Generate a single conversation based on the configuration.

        Args:
            config: Configuration for the conversation

        Returns:
            Generated conversation

        Raises:
            Exception: If generation fails
        """
        try:
            logger.info(f"Generating conversation for industry: {config.industry}, topic: {config.topics}")

            # Get persona from diversity engine if available
            persona = None
            if self.diversity_engine:
                persona = self.diversity_engine.get_persona_for_conversation()

            # Select a random topic if multiple provided
            topic = random.choice(config.topics) if config.topics else "general inquiry"

            # Build prompt for Claude
            prompt = self._build_prompt(config, topic, persona)

            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=1.0,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response with config for role names
            conversation_text = response.content[0].text
            messages = self._parse_conversation(conversation_text, config.message_count, config)

            # Create conversation object with full metadata including domain
            metadata = ConversationMetadata(
                domain=config.domain,
                industry=config.industry,
                topic=topic,
                scenario=config.scenario,
                role_1=config.role_1,
                role_2=config.role_2,
                tone=config.tone,
                message_count=len(messages),
            )

            # Add persona to metadata if available
            if persona:
                metadata.persona = persona.to_dict()

            conversation = Conversation(
                id=str(uuid.uuid4()),
                messages=messages,
                metadata=metadata
            )

            logger.info(f"Successfully generated conversation {conversation.id}")
            return conversation

        except Exception as e:
            logger.error(f"Failed to generate conversation: {e}")
            raise

    def generate_batch(self, config: ConversationConfig, count: int) -> List[Conversation]:
        """
        Generate multiple conversations.

        Args:
            config: Configuration for the conversations
            count: Number of conversations to generate

        Returns:
            List of generated conversations
        """
        conversations = []
        for i in range(count):
            try:
                logger.info(f"Generating conversation {i+1}/{count}")
                conversation = self.generate(config)
                conversations.append(conversation)
            except Exception as e:
                logger.error(f"Failed to generate conversation {i+1}: {e}")
                continue

        logger.info(f"Successfully generated {len(conversations)}/{count} conversations")
        return conversations

    def _build_prompt(
        self,
        config: ConversationConfig,
        topic: str,
        persona: Optional[Persona] = None
    ) -> str:
        """Build the prompt for Claude to generate a domain-agnostic conversation."""
        # Get domain context
        domain_context = DomainVocabulary.build_context_hint(
            config.domain,
            config.role_1,
            config.role_2,
            config.scenario
        )

        role_1_name = config.role_1_name or f"the {config.role_1}"
        role_2_name = config.role_2_name or f"the {config.role_2}"

        # Get role descriptions
        role_1_desc = DomainVocabulary.get_role_description(config.domain, config.role_1)
        role_2_desc = DomainVocabulary.get_role_description(config.domain, config.role_2)

        # Build base prompt with domain-agnostic language
        scenario_text = config.scenario if config.scenario else topic

        prompt = f"""Generate a realistic conversation in the following context:

{domain_context}

Specific Context:
Industry: {config.industry}
Topic/Scenario: {scenario_text}
Tone: {config.tone}
Number of messages: {config.message_count} (must alternate between {config.role_1} and {config.role_2}, starting with {config.role_1})

Requirements:
- Create a natural, realistic interaction appropriate for this domain
- {config.role_1.capitalize()} identifier: {role_1_name}
- {config.role_2.capitalize()} identifier: {role_2_name}
- The conversation should feel authentic with appropriate language for the {config.tone} tone
- Include realistic details, terminology, and context relevant to {config.industry} and this domain
- Messages should flow naturally and build on previous context
- {config.role_2.capitalize()} should be {DomainVocabulary.get_tone_guidance(config.domain).split(',')[0].lower()}
"""

        # Add persona-specific guidance if available
        if persona:
            prompt += f"""
{config.role_1.capitalize()} Persona Characteristics:
- Age range: {persona.age_range}
- Communication style: {persona.communication_style}
- Tech savviness: {persona.tech_savviness}
- Patience level: {persona.patience_level}
- Vocabulary level: {persona.vocabulary_level}
- The {config.role_1} should exhibit these traits naturally through their messages
"""
            if persona.typical_phrases:
                prompt += f"- May use phrases like: {', '.join(persona.typical_phrases[:2])}\n"

            if persona.emoji_usage != "none":
                prompt += f"- Emoji usage: {persona.emoji_usage}\n"

        # Add style guidance from diversity engine if available
        if self.diversity_engine and self.diversity_engine.style_pattern:
            style_guidance = self.diversity_engine.get_style_guidance()
            if style_guidance:
                prompt += f"\n{style_guidance}"

        # Add format instructions with dynamic role names
        role_1_upper = config.role_1.upper()
        role_2_upper = config.role_2.upper()

        prompt += f"""
Format your response as a series of messages, one per line, in this exact format:
{role_1_upper}: [message text]
{role_2_upper}: [message text]

Start with the {config.role_1}'s initial message and alternate strictly between {role_1_upper} and {role_2_upper}.
Generate exactly {{message_count}} messages total.

Example format:
{role_1_upper}: [First message appropriate for this domain and role]
{role_2_upper}: [Response appropriate for this domain and role]

Now generate the conversation:"""

        prompt = prompt.format(message_count=config.message_count)
        return prompt

    def _parse_conversation(self, text: str, expected_count: int, config: ConversationConfig) -> List[Message]:
        """
        Parse the generated conversation text into Message objects.

        Args:
            text: Raw conversation text from Claude
            expected_count: Expected number of messages
            config: Conversation configuration with role names

        Returns:
            List of Message objects
        """
        messages = []
        lines = text.strip().split('\n')

        # Generate timestamps with realistic intervals (30 seconds to 3 minutes between messages)
        base_time = datetime.now() - timedelta(hours=1)
        current_time = base_time

        # Dynamic role prefixes
        role_1_prefix = f"{config.role_1.upper()}:"
        role_2_prefix = f"{config.role_2.upper()}:"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse dynamic role prefixes
            if line.startswith(role_1_prefix):
                role = config.role_1
                content = line[len(role_1_prefix):].strip()
            elif line.startswith(role_2_prefix):
                role = config.role_2
                content = line[len(role_2_prefix):].strip()
            else:
                # Try backwards compatibility with CUSTOMER/AGENT for existing seeds
                if line.startswith("CUSTOMER:"):
                    role = config.role_1  # Map CUSTOMER to role_1
                    content = line[9:].strip()
                elif line.startswith("AGENT:"):
                    role = config.role_2  # Map AGENT to role_2
                    content = line[6:].strip()
                else:
                    # Skip lines that don't match any format
                    continue

            if content:
                messages.append(Message(
                    role=role,
                    content=content,
                    timestamp=current_time
                ))
                # Add random time interval for next message
                current_time += timedelta(seconds=random.randint(30, 180))

        # Validate we got the expected number of messages
        if len(messages) != expected_count:
            logger.warning(
                f"Expected {expected_count} messages but parsed {len(messages)}. "
                "Adjusting..."
            )

        # Ensure we have at least 2 messages
        if len(messages) < 2:
            raise ValueError("Failed to parse valid conversation from response")

        return messages[:expected_count] if len(messages) >= expected_count else messages
