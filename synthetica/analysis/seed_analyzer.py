"""
Seed conversation analyzer for extracting style patterns and detecting domain.
"""
import logging
import re
import statistics
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional
import random

from synthetica.generators.domain_vocabulary import DomainVocabulary

logger = logging.getLogger(__name__)


class SeedAnalyzer:
    """Analyzes seed conversations to extract style patterns and generate personas."""

    def __init__(self):
        """Initialize the seed analyzer."""
        self.formality_indicators = {
            'formal': [
                r'\b(kindly|regards|sincerely|furthermore|therefore|nevertheless)\b',
                r'\b(professional|appreciate|apologize|understand)\b',
            ],
            'informal': [
                r'\b(hey|yeah|yep|nope|gonna|wanna|gotta)\b',
                r'\b(cool|awesome|super|great|thanks|thx)\b',
            ]
        }

        self.sentiment_indicators = {
            'positive': [
                r'\b(thank|thanks|great|excellent|wonderful|perfect|appreciate)\b',
                r'\b(happy|glad|pleased|satisfied|amazing)\b',
            ],
            'negative': [
                r'\b(unfortunately|sorry|issue|problem|broken|error|wrong)\b',
                r'\b(frustrated|upset|disappointed|concerned)\b',
            ]
        }

    def detect_domain(self, conversations: List[Dict[str, Any]]) -> Tuple[str, Dict[str, int]]:
        """
        Detect the most likely domain based on vocabulary analysis.

        Args:
            conversations: List of conversation dictionaries

        Returns:
            Tuple of (detected_domain, domain_scores)
        """
        if not conversations:
            return "customer_support", {}

        # Collect all message content
        all_text = []
        for conv in conversations:
            messages = conv.get('messages', [])
            for msg in messages:
                content = msg.get('content', '').lower()
                all_text.append(content)

        combined_text = ' '.join(all_text)

        # Score each domain based on vocabulary matches
        domain_scores = {}
        supported_domains = DomainVocabulary.get_supported_domains()

        for domain in supported_domains:
            vocabulary = DomainVocabulary.get_vocabulary_hints(domain)
            score = 0

            # Count vocabulary matches
            for vocab_word in vocabulary:
                # Use word boundary regex to match whole words
                pattern = r'\b' + re.escape(vocab_word.lower()) + r'\b'
                matches = len(re.findall(pattern, combined_text))
                score += matches

            domain_scores[domain] = score

        # Find domain with highest score
        if not domain_scores or all(score == 0 for score in domain_scores.values()):
            logger.warning("No domain vocabulary matches found, defaulting to customer_support")
            return "customer_support", domain_scores

        detected_domain = max(domain_scores, key=domain_scores.get)
        logger.info(f"Detected domain: {detected_domain} (score: {domain_scores[detected_domain]})")

        return detected_domain, domain_scores

    def extract_roles(self, conversations: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the two primary roles from seed conversations.

        Args:
            conversations: List of conversation dictionaries

        Returns:
            Tuple of (role_1, role_2) or (None, None) if cannot determine
        """
        if not conversations:
            return None, None

        # Collect all unique roles
        role_counts = Counter()

        for conv in conversations:
            messages = conv.get('messages', [])
            for msg in messages:
                role = msg.get('role', '').lower().strip()
                if role:
                    role_counts[role] += 1

        # Get the two most common roles
        most_common = role_counts.most_common(2)

        if len(most_common) < 2:
            logger.warning("Could not detect two distinct roles from seeds")
            return None, None

        role_1 = most_common[0][0]
        role_2 = most_common[1][0]

        logger.info(f"Detected roles: {role_1} ({most_common[0][1]} messages), {role_2} ({most_common[1][1]} messages)")

        return role_1, role_2

    def analyze_seeds(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a collection of seed conversations.

        Args:
            conversations: List of conversation dictionaries with 'messages' field

        Returns:
            Dictionary with analysis results
        """
        if not conversations:
            return self._get_default_analysis()

        logger.info(f"Analyzing {len(conversations)} seed conversations")

        # Detect domain and roles from conversations
        detected_domain, domain_scores = self.detect_domain(conversations)
        role_1, role_2 = self.extract_roles(conversations)

        # If roles not detected, use domain defaults
        if not role_1 or not role_2:
            role_1, role_2 = DomainVocabulary.get_default_roles(detected_domain)
            logger.info(f"Using default roles for {detected_domain}: {role_1}, {role_2}")

        all_messages = []
        role_1_messages = []
        role_2_messages = []

        # Collect all messages with dynamic role detection
        for conv in conversations:
            messages = conv.get('messages', [])
            all_messages.extend(messages)

            for msg in messages:
                role = msg.get('role', '').lower()
                content = msg.get('content', '')

                # Match against detected roles (or fallback to customer/agent for backwards compatibility)
                if role == role_1 or (role == 'customer' and role_1 == 'customer'):
                    role_1_messages.append(content)
                elif role == role_2 or (role == 'agent' and role_2 == 'agent'):
                    role_2_messages.append(content)

        # Analyze different aspects
        analysis = {
            'domain': detected_domain,
            'domain_scores': domain_scores,
            'role_1': role_1,
            'role_2': role_2,
            'message_stats': self._analyze_message_stats(all_messages, role_1_messages, role_2_messages),
            'vocabulary': self._analyze_vocabulary(role_1_messages),
            'formality': self._analyze_formality(role_1_messages),
            'sentiment': self._analyze_sentiment(role_1_messages),
            'emoji_usage': self._analyze_emoji_usage(role_1_messages),
            'typo_patterns': self._analyze_typo_patterns(role_1_messages),
            'common_phrases': self._extract_common_phrases(role_1_messages),
            'sentence_patterns': self._analyze_sentence_patterns(role_1_messages),
            'conversation_count': len(conversations),
        }

        logger.info(f"Seed analysis complete: domain={detected_domain}, roles={role_1}/{role_2}")
        return analysis

    def _analyze_message_stats(
        self,
        all_messages: List[Dict],
        role_1_messages: List[str],
        role_2_messages: List[str]
    ) -> Dict[str, Any]:
        """Analyze message length statistics."""
        role_1_lengths = [len(msg.split()) for msg in role_1_messages if msg]
        role_2_lengths = [len(msg.split()) for msg in role_2_messages if msg]

        return {
            'avg_role_1_length': round(statistics.mean(role_1_lengths), 1) if role_1_lengths else 0,
            'avg_role_2_length': round(statistics.mean(role_2_lengths), 1) if role_2_lengths else 0,
            'role_1_length_range': [
                min(role_1_lengths) if role_1_lengths else 0,
                max(role_1_lengths) if role_1_lengths else 0
            ],
            'total_messages': len(all_messages),
            'role_1_messages': len(role_1_messages),
            'role_2_messages': len(role_2_messages),
            # Backwards compatibility aliases
            'avg_customer_length': round(statistics.mean(role_1_lengths), 1) if role_1_lengths else 0,
            'avg_agent_length': round(statistics.mean(role_2_lengths), 1) if role_2_lengths else 0,
            'customer_length_range': [
                min(role_1_lengths) if role_1_lengths else 0,
                max(role_1_lengths) if role_1_lengths else 0
            ],
            'customer_messages': len(role_1_messages),
            'agent_messages': len(role_2_messages),
        }

    def _analyze_vocabulary(self, messages: List[str]) -> Dict[str, Any]:
        """Analyze vocabulary usage."""
        all_words = []
        for msg in messages:
            words = re.findall(r'\b\w+\b', msg.lower())
            all_words.extend(words)

        if not all_words:
            return {'unique_word_ratio': 0, 'total_words': 0, 'unique_words': 0, 'common_words': []}

        word_counts = Counter(all_words)
        unique_ratio = len(set(all_words)) / len(all_words)

        # Get most common words (excluding very common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'i', 'you', 'my'}
        common_words = [word for word, count in word_counts.most_common(30) if word not in stop_words][:10]

        return {
            'unique_word_ratio': round(unique_ratio, 3),
            'total_words': len(all_words),
            'unique_words': len(set(all_words)),
            'common_words': common_words,
            'vocabulary_level': self._assess_vocabulary_level(all_words)
        }

    def _assess_vocabulary_level(self, words: List[str]) -> str:
        """Assess vocabulary sophistication level."""
        avg_word_length = statistics.mean([len(w) for w in words]) if words else 0

        # Simple heuristic based on word length
        if avg_word_length < 4:
            return 'basic'
        elif avg_word_length < 5.5:
            return 'intermediate'
        else:
            return 'advanced'

    def _analyze_formality(self, messages: List[str]) -> Dict[str, Any]:
        """Analyze formality level."""
        formal_count = 0
        informal_count = 0

        for msg in messages:
            msg_lower = msg.lower()

            for pattern in self.formality_indicators['formal']:
                formal_count += len(re.findall(pattern, msg_lower))

            for pattern in self.formality_indicators['informal']:
                informal_count += len(re.findall(pattern, msg_lower))

        total_indicators = formal_count + informal_count

        if total_indicators == 0:
            formality_score = 0.5
        else:
            formality_score = formal_count / total_indicators

        # Classify
        if formality_score > 0.6:
            formality_level = 'formal'
        elif formality_score < 0.4:
            formality_level = 'informal'
        else:
            formality_level = 'neutral'

        return {
            'formality_level': formality_level,
            'formality_score': round(formality_score, 2),
            'formal_indicators': formal_count,
            'informal_indicators': informal_count,
        }

    def _analyze_sentiment(self, messages: List[str]) -> Dict[str, Any]:
        """Analyze sentiment patterns."""
        positive_count = 0
        negative_count = 0

        for msg in messages:
            msg_lower = msg.lower()

            for pattern in self.sentiment_indicators['positive']:
                positive_count += len(re.findall(pattern, msg_lower))

            for pattern in self.sentiment_indicators['negative']:
                negative_count += len(re.findall(pattern, msg_lower))

        total_indicators = positive_count + negative_count

        if total_indicators == 0:
            sentiment_score = 0.5
        else:
            sentiment_score = positive_count / total_indicators

        return {
            'sentiment_score': round(sentiment_score, 2),
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
        }

    def _analyze_emoji_usage(self, messages: List[str]) -> Dict[str, Any]:
        """Analyze emoji usage patterns."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )

        emoji_count = 0
        messages_with_emoji = 0

        for msg in messages:
            emojis = emoji_pattern.findall(msg)
            if emojis:
                emoji_count += len(emojis)
                messages_with_emoji += 1

        total_messages = len(messages) if messages else 1

        emoji_frequency = messages_with_emoji / total_messages

        # Classify usage
        if emoji_frequency > 0.3:
            usage_level = 'frequent'
        elif emoji_frequency > 0.1:
            usage_level = 'occasional'
        else:
            usage_level = 'rare'

        return {
            'emoji_usage_level': usage_level,
            'emoji_frequency': round(emoji_frequency, 2),
            'total_emojis': emoji_count,
            'messages_with_emoji': messages_with_emoji,
        }

    def _analyze_typo_patterns(self, messages: List[str]) -> Dict[str, Any]:
        """Detect intentional typo patterns (not actual spelling errors)."""
        # Look for common intentional typos/shortcuts
        typo_patterns = [
            r'\b(ur|u|r)\b',  # your, you, are
            r'\b(plz|pls)\b',  # please
            r'\b(thx|ty)\b',  # thanks, thank you
            r'\b(idk|tbh|imo)\b',  # common abbreviations
        ]

        typo_count = 0
        for msg in messages:
            msg_lower = msg.lower()
            for pattern in typo_patterns:
                typo_count += len(re.findall(pattern, msg_lower))

        total_words = sum(len(msg.split()) for msg in messages)
        typo_frequency = typo_count / total_words if total_words > 0 else 0

        return {
            'typo_frequency': round(typo_frequency, 3),
            'typo_count': typo_count,
            'uses_shortcuts': typo_frequency > 0.01
        }

    def _extract_common_phrases(self, messages: List[str], n_phrases: int = 10) -> List[str]:
        """Extract common 2-3 word phrases."""
        all_text = ' '.join(messages).lower()

        # Extract 2-word phrases
        words = re.findall(r'\b\w+\b', all_text)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]

        # Count and get most common
        phrase_counts = Counter(bigrams)

        # Filter out very common but meaningless phrases
        stop_phrases = {'i am', 'it is', 'that is', 'this is', 'to the', 'in the', 'of the'}
        common_phrases = [
            phrase for phrase, count in phrase_counts.most_common(n_phrases * 3)
            if phrase not in stop_phrases and count > 1
        ][:n_phrases]

        return common_phrases

    def _analyze_sentence_patterns(self, messages: List[str]) -> Dict[str, Any]:
        """Analyze sentence structure patterns."""
        sentence_lengths = []
        question_count = 0
        exclamation_count = 0

        for msg in messages:
            sentences = re.split(r'[.!?]+', msg)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    sentence_lengths.append(len(sentence.split()))

            question_count += msg.count('?')
            exclamation_count += msg.count('!')

        avg_sentence_length = statistics.mean(sentence_lengths) if sentence_lengths else 0
        sentence_length_variance = statistics.variance(sentence_lengths) if len(sentence_lengths) > 1 else 0

        return {
            'avg_sentence_length': round(avg_sentence_length, 1),
            'sentence_length_variance': round(sentence_length_variance, 1),
            'question_frequency': question_count,
            'exclamation_frequency': exclamation_count,
        }

    def generate_personas_from_analysis(
        self,
        analysis: Dict[str, Any],
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic personas based on seed analysis.

        Args:
            analysis: Seed analysis results
            count: Number of personas to generate (5-10)

        Returns:
            List of persona dictionaries
        """
        logger.info(f"Generating {count} personas from seed analysis")

        personas = []

        # Base characteristics from analysis
        base_formality = analysis['formality']['formality_level']
        base_emoji_usage = analysis['emoji_usage']['emoji_usage_level']
        base_vocab_level = analysis['vocabulary']['vocabulary_level']

        # Generate varied personas
        for i in range(count):
            persona = self._create_persona_variant(
                i,
                base_formality,
                base_emoji_usage,
                base_vocab_level,
                analysis
            )
            personas.append(persona)

        logger.info(f"Generated {len(personas)} personas")
        return personas

    def _create_persona_variant(
        self,
        index: int,
        base_formality: str,
        base_emoji_usage: str,
        base_vocab_level: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a persona variant with slight variations from base."""

        # Vary formality slightly
        formality_options = ['formal', 'neutral', 'informal']
        base_idx = formality_options.index(base_formality) if base_formality in formality_options else 1

        # Create variations
        if index % 3 == 0 and base_idx > 0:
            formality = formality_options[base_idx - 1]
        elif index % 3 == 1 and base_idx < len(formality_options) - 1:
            formality = formality_options[base_idx + 1]
        else:
            formality = base_formality

        # Vary emoji usage
        emoji_options = ['rare', 'occasional', 'frequent']
        emoji_idx = emoji_options.index(base_emoji_usage) if base_emoji_usage in emoji_options else 1

        if index % 2 == 0 and emoji_idx > 0:
            emoji_usage = emoji_options[emoji_idx - 1]
        else:
            emoji_usage = base_emoji_usage

        # Create persona
        persona = {
            'id': f'seed_persona_{index + 1}',
            'name': f'Seed Persona {index + 1}',
            'communication_style': formality,
            'vocabulary_level': base_vocab_level,
            'emoji_usage': emoji_usage,
            'tech_savviness': random.choice(['low', 'medium', 'high']),
            'patience_level': random.choice(['low', 'medium', 'high']),
            'age_range': random.choice(['18-24', '25-34', '35-44', '45-54', '55+']),
            'typical_phrases': random.sample(
                analysis['common_phrases'],
                min(3, len(analysis['common_phrases']))
            ) if analysis['common_phrases'] else [],
            'message_length_preference': self._get_length_preference(
                analysis['message_stats']['avg_customer_length']
            ),
            'uses_shortcuts': analysis['typo_patterns']['uses_shortcuts'] and random.random() > 0.5,
        }

        return persona

    def _get_length_preference(self, avg_length: float) -> str:
        """Determine message length preference."""
        if avg_length < 10:
            return 'short'
        elif avg_length < 25:
            return 'medium'
        else:
            return 'long'

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when no seeds provided."""
        return {
            'domain': 'customer_support',
            'domain_scores': {},
            'role_1': 'customer',
            'role_2': 'agent',
            'message_stats': {
                'avg_role_1_length': 15,
                'avg_role_2_length': 20,
                'role_1_length_range': [5, 30],
                'total_messages': 0,
                'role_1_messages': 0,
                'role_2_messages': 0,
                # Backwards compatibility
                'avg_customer_length': 15,
                'avg_agent_length': 20,
                'customer_length_range': [5, 30],
                'customer_messages': 0,
                'agent_messages': 0,
            },
            'vocabulary': {
                'unique_word_ratio': 0.5,
                'total_words': 0,
                'unique_words': 0,
                'common_words': [],
                'vocabulary_level': 'intermediate'
            },
            'formality': {
                'formality_level': 'neutral',
                'formality_score': 0.5,
                'formal_indicators': 0,
                'informal_indicators': 0,
            },
            'sentiment': {
                'sentiment_score': 0.5,
                'positive_indicators': 0,
                'negative_indicators': 0,
            },
            'emoji_usage': {
                'emoji_usage_level': 'occasional',
                'emoji_frequency': 0.1,
                'total_emojis': 0,
                'messages_with_emoji': 0,
            },
            'typo_patterns': {
                'typo_frequency': 0.0,
                'typo_count': 0,
                'uses_shortcuts': False
            },
            'common_phrases': [],
            'sentence_patterns': {
                'avg_sentence_length': 12,
                'sentence_length_variance': 10,
                'question_frequency': 0,
                'exclamation_frequency': 0,
            },
            'conversation_count': 0,
        }
