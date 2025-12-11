"""
Pydantic schemas for conversation data models.
"""
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """A single message in a conversation."""
    role: Literal["customer", "agent"]
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
    industry: str
    topic: str
    tone: Literal["formal", "casual", "empathetic", "professional"]
    message_count: int
    generated_at: datetime = Field(default_factory=datetime.now)
    persona: Optional[Dict[str, Any]] = Field(None, description="Customer persona characteristics")


class Conversation(BaseModel):
    """A complete customer support conversation."""
    id: str
    messages: List[Message] = Field(..., min_length=2)
    metadata: ConversationMetadata
    quality_score: Optional[float] = Field(None, ge=0, le=100)

    @field_validator("messages")
    @classmethod
    def validate_message_alternation(cls, v: List[Message]) -> List[Message]:
        """Ensure messages alternate between customer and agent."""
        if len(v) < 2:
            raise ValueError("Conversation must have at least 2 messages")

        # First message should be from customer
        if v[0].role != "customer":
            raise ValueError("First message must be from customer")

        # Check alternation
        for i in range(1, len(v)):
            if v[i].role == v[i-1].role:
                raise ValueError(f"Messages must alternate between customer and agent at position {i}")

        return v


class ConversationConfig(BaseModel):
    """Configuration for conversation generation."""
    industry: str = Field(default="technology", description="Industry context for the conversation")
    topics: List[str] = Field(default_factory=lambda: ["general inquiry"], description="List of possible topics")
    tone: Literal["formal", "casual", "empathetic", "professional"] = Field(default="professional")
    message_count: int = Field(default=8, ge=4, le=20, description="Number of messages to generate")
    customer_name: Optional[str] = Field(None, description="Optional customer name")
    company_name: Optional[str] = Field(None, description="Optional company name")

    @field_validator("message_count")
    @classmethod
    def message_count_must_be_even(cls, v: int) -> int:
        """Ensure message count is even so conversations end with agent response."""
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
