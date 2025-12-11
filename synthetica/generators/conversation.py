"""
Customer support conversation generator using Claude API.
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

            # Parse response
            conversation_text = response.content[0].text
            messages = self._parse_conversation(conversation_text, config.message_count)

            # Create conversation object with persona metadata
            metadata = ConversationMetadata(
                industry=config.industry,
                topic=topic,
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
        """Build the prompt for Claude to generate a conversation."""
        customer_name = config.customer_name or "the customer"
        company_name = config.company_name or "the company"

        # Build base prompt
        prompt = f"""Generate a realistic customer support conversation in the {config.industry} industry.

Topic: {topic}
Tone: {config.tone}
Number of messages: {config.message_count} (must alternate between customer and agent, starting with customer)

Requirements:
- Create a natural, realistic customer support interaction
- Customer name: {customer_name}
- Company: {company_name}
- The conversation should feel authentic with appropriate language for the {config.tone} tone
- Include realistic details, product names, issues, and resolutions relevant to {config.industry}
- Messages should flow naturally and build on previous context
- Agent should be helpful and professional
"""

        # Add persona-specific guidance if available
        if persona:
            prompt += f"""
Customer Persona Characteristics:
- Age range: {persona.age_range}
- Communication style: {persona.communication_style}
- Tech savviness: {persona.tech_savviness}
- Patience level: {persona.patience_level}
- Vocabulary level: {persona.vocabulary_level}
- The customer should exhibit these traits naturally through their messages
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

        # Add format instructions
        prompt += """
Format your response as a series of messages, one per line, in this exact format:
CUSTOMER: [message text]
AGENT: [message text]

Start with the customer's initial message and alternate strictly between CUSTOMER and AGENT.
Generate exactly {message_count} messages total.

Example format:
CUSTOMER: Hi, I'm having trouble accessing my account.
AGENT: Hello! I'd be happy to help you with that. Can you tell me what error message you're seeing?
CUSTOMER: It says "Invalid credentials" even though I'm sure my password is correct.
AGENT: I understand how frustrating that can be. Let me check your account status...

Now generate the conversation:"""

        prompt = prompt.format(message_count=config.message_count)
        return prompt

    def _parse_conversation(self, text: str, expected_count: int) -> List[Message]:
        """
        Parse the generated conversation text into Message objects.

        Args:
            text: Raw conversation text from Claude
            expected_count: Expected number of messages

        Returns:
            List of Message objects
        """
        messages = []
        lines = text.strip().split('\n')

        # Generate timestamps with realistic intervals (30 seconds to 3 minutes between messages)
        base_time = datetime.now() - timedelta(hours=1)
        current_time = base_time

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse CUSTOMER: or AGENT: prefix
            if line.startswith("CUSTOMER:"):
                role = "customer"
                content = line[9:].strip()
            elif line.startswith("AGENT:"):
                role = "agent"
                content = line[6:].strip()
            else:
                # Skip lines that don't match format
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
