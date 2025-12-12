"""
Pydantic schemas for conversation data models.
"""
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator


# Domain definitions
SUPPORTED_DOMAINS = Literal[
    "customer_support",
    "healthcare",
    "sales",
    "education",
    "legal",
    "recruiting",
    "financial_services",
    "real_estate"
]


class Message(BaseModel):
    """A single message in a conversation."""
    role: str  # Changed from Literal to flexible string (e.g., customer, patient, student, client)
    content: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v


class ConversationMetadata(BaseModel):
    """Metadata about a conversation."""
    domain: str = Field(default="customer_support", description="Domain of the conversation")
    industry: str = Field(default="technology", description="Industry context")
    topic: str = Field(description="Topic of the conversation")
    scenario: Optional[str] = Field(None, description="Specific scenario being discussed")
    role_1: str = Field(default="customer", description="First role in conversation")
    role_2: str = Field(default="agent", description="Second role in conversation")
    tone: Literal["formal", "casual", "empathetic", "professional"] = Field(default="professional")
    message_count: int
    generated_at: datetime = Field(default_factory=datetime.now)
    persona: Optional[Dict[str, Any]] = Field(None, description="Persona characteristics for role_1")


class Conversation(BaseModel):
    """A complete conversation between two roles in any domain."""
    id: str
    messages: List[Message] = Field(..., min_length=2)
    metadata: ConversationMetadata
    quality_score: Optional[float] = Field(None, ge=0, le=100)

    @field_validator("messages")
    @classmethod
    def validate_message_alternation(cls, v: List[Message], info) -> List[Message]:
        """Ensure messages alternate between the two roles."""
        if len(v) < 2:
            raise ValueError("Conversation must have at least 2 messages")

        # Get expected roles from metadata if available
        # Default to customer/agent for backwards compatibility
        metadata = info.data.get('metadata')
        if metadata and hasattr(metadata, 'role_1') and hasattr(metadata, 'role_2'):
            role_1 = metadata.role_1
            role_2 = metadata.role_2
        else:
            # Try to infer from messages
            role_1 = v[0].role
            # Find the second unique role
            role_2 = None
            for msg in v[1:]:
                if msg.role != role_1:
                    role_2 = msg.role
                    break

            if not role_2:
                raise ValueError("Could not determine second role in conversation")

        # First message should be from role_1
        if v[0].role != role_1:
            raise ValueError(f"First message must be from {role_1}")

        # Check alternation
        for i in range(1, len(v)):
            if v[i].role == v[i-1].role:
                raise ValueError(f"Messages must alternate between {role_1} and {role_2} at position {i}")

            # Validate role is either role_1 or role_2
            if v[i].role not in [role_1, role_2]:
                raise ValueError(f"Invalid role '{v[i].role}' at position {i}. Expected {role_1} or {role_2}")

        return v


class ConversationConfig(BaseModel):
    """Configuration for conversation generation."""
    domain: str = Field(default="customer_support", description="Domain of the conversation")
    industry: str = Field(default="technology", description="Industry context for the conversation")
    topics: List[str] = Field(default_factory=lambda: ["general inquiry"], description="List of possible topics")
    scenario: Optional[str] = Field(None, description="Specific scenario being discussed")
    role_1: str = Field(default="customer", description="First role in conversation (initiates)")
    role_2: str = Field(default="agent", description="Second role in conversation (responds)")
    tone: Literal["formal", "casual", "empathetic", "professional"] = Field(default="professional")
    message_count: int = Field(default=8, ge=4, le=20, description="Number of messages to generate")
    role_1_name: Optional[str] = Field(None, description="Optional name for role_1 (e.g., customer name, patient name)")
    role_2_name: Optional[str] = Field(None, description="Optional name for role_2 (e.g., company name, doctor name)")

    # Backwards compatibility aliases
    @property
    def customer_name(self) -> Optional[str]:
        """Backwards compatibility alias for role_1_name."""
        return self.role_1_name

    @property
    def company_name(self) -> Optional[str]:
        """Backwards compatibility alias for role_2_name."""
        return self.role_2_name

    @field_validator("message_count")
    @classmethod
    def message_count_must_be_even(cls, v: int) -> int:
        """Ensure message count is even so conversations end with role_2 response."""
        if v % 2 != 0:
            v += 1
        return v


class QualityScore(BaseModel):
    """Quality scores for a conversation."""
    coherence: float = Field(..., ge=0, le=100, description="How logically connected the conversation is")
    diversity: float = Field(..., ge=0, le=100, description="Variety in language and content")
    naturalness: float = Field(..., ge=0, le=100, description="How human-like the conversation feels")
    overall: float = Field(..., ge=0, le=100, description="Overall quality score")

    @field_validator("overall")
    @classmethod
    def calculate_overall(cls, v: float, info) -> float:
        """Calculate overall score as average of component scores."""
        if "coherence" in info.data and "diversity" in info.data and "naturalness" in info.data:
            return round((info.data["coherence"] + info.data["diversity"] + info.data["naturalness"]) / 3, 2)
        return v
