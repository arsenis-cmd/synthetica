"""
Diversity engine for generating varied and realistic customer personas and conversations.
"""
import re
import random
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
import difflib


@dataclass
class Persona:
    """Represents a customer persona with distinct characteristics."""

    id: str
    name: str
    age_range: str
    tone: str
    vocabulary_level: str  # basic, intermediate, advanced
    patience_level: str  # low, medium, high
    tech_savviness: str  # beginner, intermediate, expert
    communication_style: str
    typical_phrases: List[str] = field(default_factory=list)
    emoji_usage: str = "none"  # none, minimal, frequent

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary for metadata."""
        return {
            "id": self.id,
            "name": self.name,
            "age_range": self.age_range,
            "tone": self.tone,
            "vocabulary_level": self.vocabulary_level,
            "patience_level": self.patience_level,
            "tech_savviness": self.tech_savviness,
            "communication_style": self.communication_style,
            "emoji_usage": self.emoji_usage
        }


class PersonaGenerator:
    """Generates and manages diverse personas for any role/domain."""

    def __init__(self):
        """Initialize with predefined personas."""
        self.personas = self._create_default_personas()
        self.domain_personas = self._create_domain_personas()

    def _create_default_personas(self) -> List[Persona]:
        """Create 12 distinct customer personas."""
        return [
            Persona(
                id="tech_millennial",
                name="Tech-Savvy Millennial",
                age_range="25-35",
                tone="casual",
                vocabulary_level="advanced",
                patience_level="low",
                tech_savviness="expert",
                communication_style="direct and concise",
                typical_phrases=[
                    "Can you guys fix this ASAP?",
                    "This should be a quick fix",
                    "I've already tried clearing cache",
                    "Is there an API I can use instead?"
                ],
                emoji_usage="minimal"
            ),
            Persona(
                id="professional_exec",
                name="Business Executive",
                age_range="40-55",
                tone="formal",
                vocabulary_level="advanced",
                patience_level="medium",
                tech_savviness="intermediate",
                communication_style="professional and detailed",
                typical_phrases=[
                    "I need to address this matter promptly",
                    "Could you please provide a detailed explanation",
                    "I appreciate your assistance",
                    "What is your expected resolution timeline?"
                ],
                emoji_usage="none"
            ),
            Persona(
                id="frustrated_senior",
                name="Frustrated Senior",
                age_range="60+",
                tone="frustrated",
                vocabulary_level="intermediate",
                patience_level="low",
                tech_savviness="beginner",
                communication_style="verbose and emotional",
                typical_phrases=[
                    "I've been trying to figure this out for hours",
                    "This is too complicated",
                    "I just want it to work like it used to",
                    "Why do you keep changing things?"
                ],
                emoji_usage="none"
            ),
            Persona(
                id="patient_learner",
                name="Patient Learner",
                age_range="30-45",
                tone="polite",
                vocabulary_level="intermediate",
                patience_level="high",
                tech_savviness="intermediate",
                communication_style="polite and inquisitive",
                typical_phrases=[
                    "Thank you for your help",
                    "Could you explain how that works?",
                    "I want to make sure I understand correctly",
                    "I'm happy to provide more information"
                ],
                emoji_usage="minimal"
            ),
            Persona(
                id="gen_z_casual",
                name="Gen Z Casual",
                age_range="18-24",
                tone="casual",
                vocabulary_level="basic",
                patience_level="medium",
                tech_savviness="expert",
                communication_style="informal and brief",
                typical_phrases=[
                    "hey so this isn't working",
                    "nvm figured it out",
                    "thx!",
                    "can u help?",
                    "lowkey annoying"
                ],
                emoji_usage="frequent"
            ),
            Persona(
                id="anxious_customer",
                name="Anxious Customer",
                age_range="25-40",
                tone="worried",
                vocabulary_level="intermediate",
                patience_level="medium",
                tech_savviness="beginner",
                communication_style="anxious and over-explaining",
                typical_phrases=[
                    "I'm really worried this won't work",
                    "I hope I didn't break anything",
                    "Is this normal?",
                    "Should I be concerned about...?"
                ],
                emoji_usage="none"
            ),
            Persona(
                id="demanding_power_user",
                name="Demanding Power User",
                age_range="30-50",
                tone="demanding",
                vocabulary_level="advanced",
                patience_level="low",
                tech_savviness="expert",
                communication_style="assertive and technical",
                typical_phrases=[
                    "This is unacceptable",
                    "I need escalation",
                    "Your documentation is incorrect",
                    "I've already done all standard troubleshooting"
                ],
                emoji_usage="none"
            ),
            Persona(
                id="friendly_enthusiast",
                name="Friendly Enthusiast",
                age_range="25-40",
                tone="enthusiastic",
                vocabulary_level="intermediate",
                patience_level="high",
                tech_savviness="intermediate",
                communication_style="friendly and positive",
                typical_phrases=[
                    "Love your product!",
                    "Thanks so much!",
                    "Really appreciate the help!",
                    "You guys are awesome!"
                ],
                emoji_usage="frequent"
            ),
            Persona(
                id="international_customer",
                name="International Customer",
                age_range="25-50",
                tone="polite",
                vocabulary_level="intermediate",
                patience_level="high",
                tech_savviness="intermediate",
                communication_style="formal with occasional grammar issues",
                typical_phrases=[
                    "Please can you help me with this issue",
                    "I am not understanding",
                    "In my country we do this differently",
                    "Thank you very much for your kind help"
                ],
                emoji_usage="minimal"
            ),
            Persona(
                id="busy_parent",
                name="Busy Parent",
                age_range="30-45",
                tone="hurried",
                vocabulary_level="basic",
                patience_level="low",
                tech_savviness="beginner",
                communication_style="rushed and distracted",
                typical_phrases=[
                    "I don't have time for this",
                    "Can you just tell me what to click?",
                    "Sorry, kid interrupted - what were you saying?",
                    "Just need a quick fix"
                ],
                emoji_usage="none"
            ),
            Persona(
                id="skeptical_researcher",
                name="Skeptical Researcher",
                age_range="25-40",
                tone="skeptical",
                vocabulary_level="advanced",
                patience_level="medium",
                tech_savviness="expert",
                communication_style="questioning and analytical",
                typical_phrases=[
                    "Can you provide documentation for this?",
                    "How does this comply with privacy regulations?",
                    "What's the underlying cause?",
                    "I need to verify this independently"
                ],
                emoji_usage="none"
            ),
            Persona(
                id="first_time_user",
                name="First-Time User",
                age_range="20-60",
                tone="uncertain",
                vocabulary_level="basic",
                patience_level="high",
                tech_savviness="beginner",
                communication_style="hesitant and seeking guidance",
                typical_phrases=[
                    "I'm new to this",
                    "Where do I start?",
                    "Is there a tutorial?",
                    "I don't want to mess anything up"
                ],
                emoji_usage="none"
            )
        ]

    def _create_domain_personas(self) -> Dict[str, Dict[str, List[Persona]]]:
        """Create domain-specific personas for different roles."""
        return {
            "healthcare": {
                "patient": [
                    Persona(
                        id="nervous_patient",
                        name="Nervous Patient",
                        age_range="25-55",
                        tone="anxious",
                        vocabulary_level="basic",
                        patience_level="low",
                        tech_savviness="beginner",
                        communication_style="worried and over-explaining",
                        typical_phrases=[
                            "I'm really concerned about this",
                            "Is this something serious?",
                            "I've been worrying about this for days",
                            "Should I be alarmed?"
                        ],
                        emoji_usage="none"
                    ),
                    Persona(
                        id="stoic_patient",
                        name="Stoic Patient",
                        age_range="40-65",
                        tone="matter-of-fact",
                        vocabulary_level="intermediate",
                        patience_level="high",
                        tech_savviness="intermediate",
                        communication_style="brief and factual",
                        typical_phrases=[
                            "Just the facts, please",
                            "I can handle it",
                            "What do I need to do?",
                            "No need to sugarcoat"
                        ],
                        emoji_usage="none"
                    ),
                    Persona(
                        id="detail_oriented_patient",
                        name="Detail-Oriented Patient",
                        age_range="30-50",
                        tone="analytical",
                        vocabulary_level="advanced",
                        patience_level="high",
                        tech_savviness="expert",
                        communication_style="thorough and questioning",
                        typical_phrases=[
                            "Can you explain the mechanism?",
                            "What are all the possible side effects?",
                            "I've been tracking my symptoms",
                            "I'd like to understand the options"
                        ],
                        emoji_usage="none"
                    )
                ],
                "doctor": [
                    Persona(
                        id="rushed_doctor",
                        name="Rushed Doctor",
                        age_range="35-50",
                        tone="hurried",
                        vocabulary_level="advanced",
                        patience_level="low",
                        tech_savviness="intermediate",
                        communication_style="efficient and directive",
                        typical_phrases=[
                            "Let's move quickly through this",
                            "In brief",
                            "I need to see another patient shortly",
                            "The key point is"
                        ],
                        emoji_usage="none"
                    ),
                    Persona(
                        id="empathetic_doctor",
                        name="Empathetic Doctor",
                        age_range="30-55",
                        tone="compassionate",
                        vocabulary_level="intermediate",
                        patience_level="high",
                        tech_savviness="intermediate",
                        communication_style="warm and reassuring",
                        typical_phrases=[
                            "I understand your concerns",
                            "Let's work through this together",
                            "It's completely normal to feel this way",
                            "I'm here to help you"
                        ],
                        emoji_usage="none"
                    ),
                    Persona(
                        id="clinical_doctor",
                        name="Clinical Doctor",
                        age_range="40-60",
                        tone="professional",
                        vocabulary_level="advanced",
                        patience_level="medium",
                        tech_savviness="expert",
                        communication_style="precise and medical",
                        typical_phrases=[
                            "Clinically speaking",
                            "The diagnosis indicates",
                            "Based on the symptoms",
                            "We should run some tests"
                        ],
                        emoji_usage="none"
                    )
                ]
            },
            "sales": {
                "prospect": [
                    Persona(
                        id="skeptical_prospect",
                        name="Skeptical Prospect",
                        age_range="35-50",
                        tone="doubtful",
                        vocabulary_level="advanced",
                        patience_level="low",
                        tech_savviness="expert",
                        communication_style="challenging and questioning",
                        typical_phrases=[
                            "I've heard that before",
                            "How is this different from competitors?",
                            "Can you prove that claim?",
                            "I'm not convinced yet"
                        ],
                        emoji_usage="none"
                    ),
                    Persona(
                        id="eager_prospect",
                        name="Eager Prospect",
                        age_range="25-40",
                        tone="enthusiastic",
                        vocabulary_level="intermediate",
                        patience_level="medium",
                        tech_savviness="intermediate",
                        communication_style="excited and fast-paced",
                        typical_phrases=[
                            "This sounds perfect!",
                            "When can we get started?",
                            "I love this feature",
                            "Our team needs this"
                        ],
                        emoji_usage="minimal"
                    )
                ],
                "sales_rep": [
                    Persona(
                        id="consultative_rep",
                        name="Consultative Rep",
                        age_range="30-45",
                        tone="advisory",
                        vocabulary_level="advanced",
                        patience_level="high",
                        tech_savviness="expert",
                        communication_style="solution-focused and patient",
                        typical_phrases=[
                            "Let me understand your needs first",
                            "Based on what you've shared",
                            "I'd recommend",
                            "Let's explore your options"
                        ],
                        emoji_usage="none"
                    ),
                    Persona(
                        id="pushy_rep",
                        name="Pushy Rep",
                        age_range="25-40",
                        tone="aggressive",
                        vocabulary_level="intermediate",
                        patience_level="low",
                        tech_savviness="intermediate",
                        communication_style="direct and persistent",
                        typical_phrases=[
                            "This deal won't last",
                            "I can offer you a discount today",
                            "Let's close this now",
                            "You don't want to miss this"
                        ],
                        emoji_usage="none"
                    )
                ]
            },
            "legal": {
                "client": [
                    Persona(
                        id="worried_client",
                        name="Worried Client",
                        age_range="30-60",
                        tone="concerned",
                        vocabulary_level="intermediate",
                        patience_level="medium",
                        tech_savviness="beginner",
                        communication_style="anxious and seeking reassurance",
                        typical_phrases=[
                            "What are my risks here?",
                            "Could this go badly?",
                            "I want to protect myself",
                            "What's the worst case scenario?"
                        ],
                        emoji_usage="none"
                    )
                ],
                "lawyer": [
                    Persona(
                        id="meticulous_lawyer",
                        name="Meticulous Lawyer",
                        age_range="35-55",
                        tone="precise",
                        vocabulary_level="advanced",
                        patience_level="high",
                        tech_savviness="intermediate",
                        communication_style="detailed and thorough",
                        typical_phrases=[
                            "We need to review every clause",
                            "Legally speaking",
                            "The precedent here",
                            "I'll need to examine this carefully"
                        ],
                        emoji_usage="none"
                    )
                ]
            }
        }

    def get_random_persona(self) -> Persona:
        """Get a random persona from the pool."""
        return random.choice(self.personas)

    def get_persona_by_id(self, persona_id: str) -> Optional[Persona]:
        """Get a specific persona by ID."""
        # Check default personas first
        for persona in self.personas:
            if persona.id == persona_id:
                return persona

        # Check domain personas
        for domain_roles in self.domain_personas.values():
            for role_personas in domain_roles.values():
                for persona in role_personas:
                    if persona.id == persona_id:
                        return persona
        return None

    def get_persona_for_domain_role(self, domain: str, role: str) -> Optional[Persona]:
        """Get a random persona for a specific domain and role."""
        if domain in self.domain_personas:
            if role in self.domain_personas[domain]:
                personas = self.domain_personas[domain][role]
                if personas:
                    return random.choice(personas)
        return None


@dataclass
class StylePattern:
    """Extracted style patterns from seed examples."""

    avg_sentence_length: float
    formality_score: float  # 0-1, where 1 is very formal
    punctuation_frequency: Dict[str, int]
    common_phrases: List[str]
    avg_word_length: float
    question_ratio: float  # percentage of sentences that are questions
    exclamation_ratio: float  # percentage of sentences with exclamations


class StyleExtractor:
    """Extracts and applies style patterns from seed conversations."""

    @staticmethod
    def extract_patterns(seed_examples: List[str]) -> StylePattern:
        """
        Extract style patterns from seed conversation examples.

        Args:
            seed_examples: List of example conversation texts

        Returns:
            StylePattern object with extracted patterns
        """
        if not seed_examples:
            return StyleExtractor._default_pattern()

        total_sentences = 0
        total_sentence_length = 0
        total_word_length = 0
        total_words = 0
        question_count = 0
        exclamation_count = 0
        punctuation_counts: Dict[str, int] = {}
        all_phrases: List[str] = []

        # Formal indicators
        formal_indicators = ['please', 'kindly', 'would you', 'could you', 'thank you', 'regards']
        casual_indicators = ['hey', 'yeah', 'nope', 'gonna', 'wanna', 'thanks', 'thx']
        formal_score = 0
        casual_score = 0

        for example in seed_examples:
            # Split into sentences
            sentences = re.split(r'[.!?]+', example)
            sentences = [s.strip() for s in sentences if s.strip()]

            for sentence in sentences:
                total_sentences += 1
                words = sentence.split()
                total_sentence_length += len(words)

                for word in words:
                    total_words += 1
                    total_word_length += len(word)

                # Check formality
                sentence_lower = sentence.lower()
                for indicator in formal_indicators:
                    if indicator in sentence_lower:
                        formal_score += 1
                for indicator in casual_indicators:
                    if indicator in sentence_lower:
                        casual_score += 1

                # Count punctuation
                if '?' in sentence:
                    question_count += 1
                    punctuation_counts['?'] = punctuation_counts.get('?', 0) + 1
                if '!' in sentence:
                    exclamation_count += 1
                    punctuation_counts['!'] = punctuation_counts.get('!', 0) + 1

                # Extract phrases (2-4 word combinations)
                words = sentence.split()
                for i in range(len(words) - 1):
                    phrase = ' '.join(words[i:i+2])
                    all_phrases.append(phrase.lower())

        # Calculate metrics
        avg_sentence_length = total_sentence_length / max(total_sentences, 1)
        avg_word_length = total_word_length / max(total_words, 1)
        question_ratio = question_count / max(total_sentences, 1)
        exclamation_ratio = exclamation_count / max(total_sentences, 1)

        # Calculate formality score (0-1)
        total_indicators = formal_score + casual_score
        formality_score = formal_score / max(total_indicators, 1) if total_indicators > 0 else 0.5

        # Get most common phrases
        phrase_counter = Counter(all_phrases)
        common_phrases = [phrase for phrase, _ in phrase_counter.most_common(10)]

        return StylePattern(
            avg_sentence_length=avg_sentence_length,
            formality_score=formality_score,
            punctuation_frequency=punctuation_counts,
            common_phrases=common_phrases,
            avg_word_length=avg_word_length,
            question_ratio=question_ratio,
            exclamation_ratio=exclamation_ratio
        )

    @staticmethod
    def _default_pattern() -> StylePattern:
        """Return default style pattern."""
        return StylePattern(
            avg_sentence_length=12.0,
            formality_score=0.5,
            punctuation_frequency={'?': 2, '!': 1},
            common_phrases=[],
            avg_word_length=5.0,
            question_ratio=0.2,
            exclamation_ratio=0.1
        )


class AntiRepetitionTracker:
    """Tracks generated content to prevent repetition."""

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize the anti-repetition tracker.

        Args:
            similarity_threshold: Minimum similarity (0-1) to flag as repetitive
        """
        self.similarity_threshold = similarity_threshold
        self.generated_phrases: Set[str] = set()
        self.generated_conversations: List[str] = []

    def add_conversation(self, conversation_text: str):
        """Add a conversation to the tracker."""
        self.generated_conversations.append(conversation_text)

        # Extract and store phrases (3-5 word sequences)
        words = conversation_text.lower().split()
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            self.generated_phrases.add(phrase)

    def check_similarity(self, new_conversation: str) -> float:
        """
        Check similarity of new conversation against existing ones.

        Args:
            new_conversation: New conversation text to check

        Returns:
            Maximum similarity score (0-1) against existing conversations
        """
        if not self.generated_conversations:
            return 0.0

        max_similarity = 0.0
        new_text_lower = new_conversation.lower()

        for existing in self.generated_conversations:
            existing_lower = existing.lower()
            similarity = difflib.SequenceMatcher(None, new_text_lower, existing_lower).ratio()
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def is_too_similar(self, new_conversation: str) -> bool:
        """
        Check if a conversation is too similar to existing ones.

        Args:
            new_conversation: New conversation text

        Returns:
            True if conversation is too similar (above threshold)
        """
        similarity = self.check_similarity(new_conversation)
        return similarity >= self.similarity_threshold

    def get_phrase_repetition_score(self, conversation: str) -> float:
        """
        Calculate phrase repetition score for a conversation.

        Args:
            conversation: Conversation text

        Returns:
            Repetition score (0-1, where 1 is highly repetitive)
        """
        words = conversation.lower().split()
        conversation_phrases = set()

        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            conversation_phrases.add(phrase)

        if not conversation_phrases:
            return 0.0

        # Calculate overlap with existing phrases
        overlap = len(conversation_phrases & self.generated_phrases)
        repetition_score = overlap / len(conversation_phrases)

        return repetition_score

    def reset(self):
        """Clear all tracked data."""
        self.generated_phrases.clear()
        self.generated_conversations.clear()


class DiversityScorer:
    """Calculates diversity scores for generated conversations."""

    @staticmethod
    def calculate_vocabulary_diversity(conversations: List[str]) -> float:
        """
        Calculate vocabulary diversity across conversations.

        Args:
            conversations: List of conversation texts

        Returns:
            Diversity score (0-1, where 1 is highly diverse)
        """
        if not conversations:
            return 0.0

        all_words = []
        for conv in conversations:
            words = re.findall(r'\b\w+\b', conv.lower())
            all_words.extend(words)

        if not all_words:
            return 0.0

        unique_words = len(set(all_words))
        total_words = len(all_words)

        # Type-token ratio (TTR) as diversity measure
        ttr = unique_words / total_words

        # Normalize to 0-1 scale (TTR typically ranges from 0.3-0.7 for good diversity)
        normalized_score = min(ttr / 0.7, 1.0)

        return normalized_score

    @staticmethod
    def calculate_structure_diversity(conversations: List[str]) -> float:
        """
        Calculate sentence structure diversity.

        Args:
            conversations: List of conversation texts

        Returns:
            Diversity score (0-1)
        """
        if not conversations:
            return 0.0

        sentence_lengths = []

        for conv in conversations:
            sentences = re.split(r'[.!?]+', conv)
            for sentence in sentences:
                words = sentence.split()
                if words:
                    sentence_lengths.append(len(words))

        if not sentence_lengths:
            return 0.0

        # Calculate coefficient of variation
        import statistics
        if len(sentence_lengths) < 2:
            return 0.5

        mean_length = statistics.mean(sentence_lengths)
        std_dev = statistics.stdev(sentence_lengths)

        if mean_length == 0:
            return 0.0

        cv = std_dev / mean_length

        # Normalize (CV of 0.3-0.5 is good diversity)
        normalized_score = min(cv / 0.5, 1.0)

        return normalized_score

    @staticmethod
    def calculate_persona_consistency(conversations: List[Dict[str, Any]]) -> float:
        """
        Check if personas are consistently distributed.

        Args:
            conversations: List of conversation objects with metadata

        Returns:
            Consistency score (0-1, where 1 is well-distributed)
        """
        if not conversations:
            return 0.0

        persona_ids = []
        for conv in conversations:
            if 'metadata' in conv and 'persona' in conv['metadata']:
                persona_ids.append(conv['metadata']['persona'].get('id', 'unknown'))

        if not persona_ids:
            return 0.5  # No persona data, neutral score

        # Calculate distribution uniformity
        persona_counts = Counter(persona_ids)
        total = len(persona_ids)
        unique_personas = len(persona_counts)

        # Expected count if perfectly distributed
        expected_count = total / unique_personas

        # Calculate chi-square-like deviation
        deviation_sum = sum((count - expected_count) ** 2 for count in persona_counts.values())
        max_deviation = (total - expected_count) ** 2 + (unique_personas - 1) * expected_count ** 2

        if max_deviation == 0:
            return 1.0

        consistency_score = 1.0 - (deviation_sum / max_deviation)

        return max(0.0, consistency_score)

    @staticmethod
    def calculate_overall_diversity(
        conversations: List[str],
        conversation_objects: List[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate overall diversity score with component breakdown.

        Args:
            conversations: List of conversation texts
            conversation_objects: Optional list of full conversation objects with metadata

        Returns:
            Dictionary with diversity metrics
        """
        vocab_diversity = DiversityScorer.calculate_vocabulary_diversity(conversations)
        structure_diversity = DiversityScorer.calculate_structure_diversity(conversations)

        persona_consistency = 0.5  # Default if no persona data
        if conversation_objects:
            persona_consistency = DiversityScorer.calculate_persona_consistency(conversation_objects)

        # Overall score is weighted average
        overall_score = (
            vocab_diversity * 0.4 +
            structure_diversity * 0.3 +
            persona_consistency * 0.3
        )

        return {
            "overall": overall_score,
            "vocabulary_diversity": vocab_diversity,
            "structure_diversity": structure_diversity,
            "persona_consistency": persona_consistency
        }


class DiversityEngine:
    """
    Main diversity engine coordinating persona generation, style injection,
    anti-repetition, and diversity scoring.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_diversity_score: float = 0.7
    ):
        """
        Initialize the diversity engine.

        Args:
            similarity_threshold: Threshold for flagging similar conversations
            min_diversity_score: Minimum acceptable diversity score
        """
        self.persona_generator = PersonaGenerator()
        self.anti_repetition = AntiRepetitionTracker(similarity_threshold)
        self.min_diversity_score = min_diversity_score
        self.style_pattern: Optional[StylePattern] = None

    def load_seed_examples(self, seed_examples: List[str]):
        """
        Load and extract style patterns from seed examples.

        Args:
            seed_examples: List of example conversation texts
        """
        self.style_pattern = StyleExtractor.extract_patterns(seed_examples)

    def get_persona_for_conversation(self) -> Persona:
        """Get a random persona for a new conversation."""
        return self.persona_generator.get_random_persona()

    def validate_conversation(self, conversation_text: str) -> Dict[str, Any]:
        """
        Validate a conversation against repetition rules.

        Args:
            conversation_text: The generated conversation text

        Returns:
            Dictionary with validation results
        """
        similarity_score = self.anti_repetition.check_similarity(conversation_text)
        repetition_score = self.anti_repetition.get_phrase_repetition_score(conversation_text)
        is_too_similar = self.anti_repetition.is_too_similar(conversation_text)

        return {
            "valid": not is_too_similar,
            "similarity_score": similarity_score,
            "repetition_score": repetition_score,
            "is_too_similar": is_too_similar
        }

    def add_conversation(self, conversation_text: str):
        """Add a conversation to the tracking system."""
        self.anti_repetition.add_conversation(conversation_text)

    def calculate_batch_diversity(
        self,
        conversations: List[str],
        conversation_objects: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate diversity score for a batch of conversations.

        Args:
            conversations: List of conversation texts
            conversation_objects: Optional full conversation objects

        Returns:
            Dictionary with diversity scores and flag if below threshold
        """
        diversity_scores = DiversityScorer.calculate_overall_diversity(
            conversations,
            conversation_objects
        )

        meets_threshold = diversity_scores["overall"] >= self.min_diversity_score

        return {
            **diversity_scores,
            "meets_threshold": meets_threshold,
            "min_threshold": self.min_diversity_score
        }

    def reset(self):
        """Reset the diversity engine state."""
        self.anti_repetition.reset()
        self.style_pattern = None

    def get_style_guidance(self) -> str:
        """
        Get style guidance text for prompt injection.

        Returns:
            Style guidance string to inject into generation prompt
        """
        if not self.style_pattern:
            return ""

        pattern = self.style_pattern

        guidance = f"""
Style Guidelines (based on provided examples):
- Target sentence length: {pattern.avg_sentence_length:.0f} words
- Formality level: {'formal' if pattern.formality_score > 0.6 else 'casual' if pattern.formality_score < 0.4 else 'neutral'}
- Average word length: {pattern.avg_word_length:.1f} characters
- Question frequency: {pattern.question_ratio * 100:.0f}% of sentences
- Exclamation usage: {'frequent' if pattern.exclamation_ratio > 0.15 else 'minimal' if pattern.exclamation_ratio < 0.05 else 'moderate'}
"""

        if pattern.common_phrases:
            guidance += f"- Common phrase patterns: {', '.join(pattern.common_phrases[:3])}\n"

        return guidance
