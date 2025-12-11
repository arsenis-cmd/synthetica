"""
Quality scoring system for synthetic conversations.
"""
import logging
import re
from typing import Dict, Optional

from anthropic import Anthropic

from synthetica.schemas.conversation import Conversation, QualityScore

logger = logging.getLogger(__name__)


class QualityScorer:
    """Scores the quality of generated conversations."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        """
        Initialize the quality scorer.

        Args:
            api_key: Optional Anthropic API key for AI-powered scoring
            model: Claude model to use for AI scoring
        """
        self.client = Anthropic(api_key=api_key) if api_key else None
        self.model = model

    def score(self, conversation: Conversation, use_ai: bool = True) -> QualityScore:
        """
        Score a conversation on multiple quality dimensions.

        Args:
            conversation: The conversation to score
            use_ai: Whether to use AI-powered scoring (requires API key)

        Returns:
            Quality score breakdown
        """
        if use_ai and self.client:
            return self._score_with_ai(conversation)
        else:
            return self._score_with_heuristics(conversation)

    def _score_with_heuristics(self, conversation: Conversation) -> QualityScore:
        """
        Score conversation using rule-based heuristics.

        Args:
            conversation: The conversation to score

        Returns:
            Quality score
        """
        coherence = self._calculate_coherence(conversation)
        diversity = self._calculate_diversity(conversation)
        naturalness = self._calculate_naturalness(conversation)

        overall = round((coherence + diversity + naturalness) / 3, 2)

        return QualityScore(
            coherence=coherence,
            diversity=diversity,
            naturalness=naturalness,
            overall=overall
        )

    def _score_with_ai(self, conversation: Conversation) -> QualityScore:
        """
        Score conversation using Claude API for more sophisticated analysis.

        Args:
            conversation: The conversation to score

        Returns:
            Quality score
        """
        try:
            # Format conversation for analysis
            conversation_text = self._format_for_analysis(conversation)

            # Create scoring prompt
            prompt = f"""Analyze the following customer support conversation and rate it on these dimensions (0-100 scale):

1. COHERENCE: How logically connected and consistent is the conversation? Do responses make sense in context?
2. DIVERSITY: How varied is the language use? Does it avoid repetitive patterns?
3. NATURALNESS: How human-like and authentic does the conversation feel?

Conversation:
{conversation_text}

Provide scores in this exact format:
COHERENCE: [score]
DIVERSITY: [score]
NATURALNESS: [score]

After the scores, briefly explain your reasoning (1-2 sentences per dimension)."""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse scores from response
            scores = self._parse_ai_scores(response.content[0].text)

            return QualityScore(
                coherence=scores["coherence"],
                diversity=scores["diversity"],
                naturalness=scores["naturalness"],
                overall=round((scores["coherence"] + scores["diversity"] + scores["naturalness"]) / 3, 2)
            )

        except Exception as e:
            logger.error(f"AI scoring failed, falling back to heuristics: {e}")
            return self._score_with_heuristics(conversation)

    def _calculate_coherence(self, conversation: Conversation) -> float:
        """Calculate coherence score based on message flow and context."""
        score = 100.0

        messages = conversation.messages

        # Check for very short messages (might indicate low quality)
        short_message_penalty = sum(1 for m in messages if len(m.content.split()) < 3)
        score -= short_message_penalty * 5

        # Check for question-answer patterns (good coherence indicator)
        question_count = sum(1 for m in messages if '?' in m.content)
        if question_count > 0:
            # Good conversations have questions
            score += min(question_count * 5, 15)

        # Check for conversation resolution (last message should be agent)
        if messages[-1].role == "agent":
            score += 10

        # Ensure score is in valid range
        return max(0, min(100, score))

    def _calculate_diversity(self, conversation: Conversation) -> float:
        """Calculate diversity score based on vocabulary and pattern variety."""
        score = 100.0

        # Combine all message content
        all_text = " ".join(m.content for m in conversation.messages).lower()
        words = re.findall(r'\b\w+\b', all_text)

        if len(words) == 0:
            return 0.0

        # Calculate unique word ratio
        unique_ratio = len(set(words)) / len(words)
        score = unique_ratio * 100

        # Bonus for varied sentence structures
        sentences = re.split(r'[.!?]+', all_text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

        if sentence_lengths:
            # More variation in sentence length = more diversity
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            score += min(variance, 15)

        return max(0, min(100, round(score, 2)))

    def _calculate_naturalness(self, conversation: Conversation) -> float:
        """Calculate naturalness score based on human-like patterns."""
        score = 100.0

        messages = conversation.messages

        # Check for natural language markers
        natural_markers = [
            r'\b(thanks|thank you|please|sorry|appreciate)\b',
            r'\b(hello|hi|hey)\b',
            r'\b(sure|okay|alright|got it)\b',
            r'\b(understand|see|got)\b'
        ]

        marker_count = 0
        for message in messages:
            content_lower = message.content.lower()
            for pattern in natural_markers:
                if re.search(pattern, content_lower):
                    marker_count += 1

        # Reward natural conversational markers
        score += min(marker_count * 3, 20)

        # Check for overly formal or robotic patterns
        robotic_patterns = [
            r'\bI apologize for any inconvenience\b',
            r'\bPlease be advised\b',
            r'\bKindly\b',
        ]

        robotic_count = 0
        for message in messages:
            for pattern in robotic_patterns:
                if re.search(pattern, message.content, re.IGNORECASE):
                    robotic_count += 1

        # Penalize overly robotic language
        score -= robotic_count * 10

        # Check for appropriate message lengths (not too short or too long)
        for message in messages:
            word_count = len(message.content.split())
            if word_count < 3:
                score -= 5
            elif word_count > 150:
                score -= 3

        return max(0, min(100, round(score, 2)))

    def _format_for_analysis(self, conversation: Conversation) -> str:
        """Format conversation for AI analysis."""
        lines = []
        for msg in conversation.messages:
            role = msg.role.upper()
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _parse_ai_scores(self, text: str) -> Dict[str, float]:
        """Parse scores from AI response."""
        scores = {}

        # Look for score patterns
        coherence_match = re.search(r'COHERENCE:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        diversity_match = re.search(r'DIVERSITY:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        naturalness_match = re.search(r'NATURALNESS:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)

        scores["coherence"] = float(coherence_match.group(1)) if coherence_match else 75.0
        scores["diversity"] = float(diversity_match.group(1)) if diversity_match else 75.0
        scores["naturalness"] = float(naturalness_match.group(1)) if naturalness_match else 75.0

        return scores
