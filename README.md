# Synthetica

A synthetic data generation platform for creating high-quality, realistic customer support conversations using Claude AI.

## Features

- **Tiered Pricing**: Free tier (100 conversations) + paid plans with Stripe integration
- **Usage Tracking**: API key-based usage tracking and rate limiting
- **Web Interface**: Beautiful, user-friendly web UI for generating conversations
- **REST API**: FastAPI-powered API for programmatic access
- **AI-Powered Generation**: Uses Claude API to generate natural, contextual conversations
- **Configurable**: Customize industry, topics, tone, and message count
- **Quality Scoring**: Automatic quality assessment on coherence, diversity, and naturalness
- **Multiple Export Formats**: Export to JSON, JSONL, CSV, and detailed CSV
- **Validation**: Pydantic models ensure data quality and consistency
- **Extensible**: Easy to add new generators and quality metrics

## Project Structure

```
synthetica/
├── synthetica/
│   ├── api/
│   │   ├── app.py             # FastAPI web server
│   │   └── templates/
│   │       └── index.html     # Web UI
│   ├── generators/
│   │   └── conversation.py    # Customer support conversation generator
│   ├── quality/
│   │   └── scorer.py          # Quality scoring system
│   ├── output/
│   │   └── formatters.py      # Export formatters (JSON, JSONL, CSV)
│   └── schemas/
│       └── conversation.py    # Pydantic data models
├── examples/
│   └── generate_conversations.py  # CLI script to generate conversations
├── start_server.py            # Web server startup script
├── requirements.txt           # Project dependencies
└── README.md
```

## Installation

1. Clone or navigate to the repository:
```bash
cd synthetica
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

## Quick Start

### Option 1: Web Interface (Recommended)

Start the web server:

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
python3 start_server.py
```

Then open your browser to http://localhost:8000

The web interface provides:
- Interactive form to configure generation settings
- Real-time conversation generation
- Quality score visualization
- Download buttons for JSON, JSONL, and CSV formats
- Dataset management and browsing

### Option 2: Command Line

Run the example script to generate 10 sample conversations:

```bash
python examples/generate_conversations.py
```

This will:
1. Generate 10 conversations across different industries
2. Score each conversation for quality
3. Export results to multiple formats in the `output/` directory

## Usage

### Basic Generation

```python
from synthetica.generators.conversation import ConversationGenerator
from synthetica.schemas.conversation import ConversationConfig

# Initialize generator
generator = ConversationGenerator(api_key="your-api-key")

# Configure conversation
config = ConversationConfig(
    industry="technology",
    topics=["software bug", "account access"],
    tone="professional",
    message_count=8,
    company_name="TechCorp"
)

# Generate conversation
conversation = generator.generate(config)
print(f"Generated {len(conversation.messages)} messages")
```

### Quality Scoring

```python
from synthetica.quality.scorer import QualityScorer

# Initialize scorer
scorer = QualityScorer(api_key="your-api-key")

# Score conversation
quality = scorer.score(conversation, use_ai=False)

print(f"Coherence: {quality.coherence}/100")
print(f"Diversity: {quality.diversity}/100")
print(f"Naturalness: {quality.naturalness}/100")
print(f"Overall: {quality.overall}/100")
```

### Exporting Data

```python
from synthetica.output.formatters import ConversationFormatter

# Export to JSON
ConversationFormatter.to_json(conversations, "output/conversations.json")

# Export to JSONL
ConversationFormatter.to_jsonl(conversations, "output/conversations.jsonl")

# Export to CSV
ConversationFormatter.to_csv(conversations, "output/conversations.csv")

# Export to all formats
ConversationFormatter.save_all_formats(
    conversations,
    base_path="output",
    name="my_conversations"
)
```

### Batch Generation

```python
# Generate multiple conversations
conversations = generator.generate_batch(config, count=50)

# Score all conversations
for conv in conversations:
    quality = scorer.score(conv, use_ai=False)
    conv.quality_score = quality.overall

# Export
ConversationFormatter.save_all_formats(conversations, "output")
```

## REST API

The Synthetica platform includes a FastAPI web server with REST endpoints.

### API Endpoints

#### POST /api/generate
Generate synthetic conversations.

**Request Body:**
```json
{
  "industry": "technology",
  "topics": ["software bug", "account access"],
  "count": 5,
  "customer_tone": "professional",
  "agent_style": "professional",
  "message_count": 8
}
```

**Response:**
```json
{
  "success": true,
  "count": 5,
  "conversations": [...],
  "average_quality": 89.6,
  "output_files": {
    "json": "/path/to/output.json",
    "jsonl": "/path/to/output.jsonl",
    "csv": "/path/to/output.csv"
  }
}
```

#### GET /api/datasets
List all generated datasets in the output folder.

**Response:**
```json
{
  "datasets": [
    {
      "filename": "technology_20231211_103324.json",
      "size_bytes": 25336,
      "created_at": "2025-12-11T10:26:10.242254",
      "format": "JSON"
    }
  ],
  "total_count": 10
}
```

#### GET /api/datasets/{filename}
Download a specific dataset file.

#### GET /health
Health check endpoint.

### Using the API

```bash
# Generate conversations
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "industry": "healthcare",
    "topics": ["appointment scheduling"],
    "count": 3,
    "customer_tone": "empathetic",
    "agent_style": "professional",
    "message_count": 6
  }'

# List datasets
curl http://localhost:8000/api/datasets

# Download a dataset
curl -O http://localhost:8000/api/datasets/healthcare_20231211_103324.json
```

API documentation is also available at http://localhost:8000/docs when the server is running.

## Configuration Options

### ConversationConfig

- **industry** (str): Industry context (e.g., "technology", "healthcare", "banking")
- **topics** (List[str]): List of possible conversation topics
- **tone** (str): Conversation tone - "formal", "casual", "empathetic", or "professional"
- **message_count** (int): Number of messages (4-20, must be even)
- **customer_name** (str, optional): Customer name to use
- **company_name** (str, optional): Company name to use

### Quality Scoring

The quality scorer evaluates conversations on three dimensions:

1. **Coherence** (0-100): How logically connected and consistent the conversation is
2. **Diversity** (0-100): Variety in language use and vocabulary
3. **Naturalness** (0-100): How human-like and authentic the conversation feels

Scoring modes:
- **Heuristic** (`use_ai=False`): Fast, rule-based scoring
- **AI-Powered** (`use_ai=True`): More sophisticated analysis using Claude

## Output Formats

### JSON
Complete conversation data with all metadata and messages.

### JSONL
One conversation per line, ideal for streaming processing.

### CSV (Standard)
One row per conversation with flattened conversation text.

### CSV (Detailed)
One row per message with full conversation context.

## Example Output

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "messages": [
    {
      "role": "customer",
      "content": "Hi, I'm having trouble logging into my account.",
      "timestamp": "2024-01-15T10:30:00"
    },
    {
      "role": "agent",
      "content": "I'd be happy to help you with that. Can you tell me what error message you're seeing?",
      "timestamp": "2024-01-15T10:31:00"
    }
  ],
  "metadata": {
    "industry": "technology",
    "topic": "account access",
    "tone": "professional",
    "message_count": 8,
    "generated_at": "2024-01-15T10:30:00"
  },
  "quality_score": 87.5
}
```

## Error Handling

The platform includes comprehensive error handling:

- API failures are logged and can be retried
- Invalid configurations raise validation errors
- Failed generations don't stop batch processing
- All errors are logged with context

## Logging

All modules use Python's built-in logging. Configure as needed:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Advanced Usage

### Custom Validation

Extend the Pydantic models for custom validation:

```python
from synthetica.schemas.conversation import Conversation

class CustomConversation(Conversation):
    @field_validator("messages")
    @classmethod
    def validate_custom_rules(cls, v):
        # Add custom validation
        return v
```

### Custom Scoring Metrics

Extend the QualityScorer class:

```python
from synthetica.quality.scorer import QualityScorer

class CustomScorer(QualityScorer):
    def _calculate_custom_metric(self, conversation):
        # Implement custom scoring logic
        pass
```

## Best Practices

1. **API Rate Limiting**: Be mindful of API rate limits when generating large batches
2. **Quality vs Speed**: Use heuristic scoring for large batches, AI scoring for critical data
3. **Configuration Variety**: Use diverse configurations to avoid repetitive patterns
4. **Validation**: Always validate generated data meets your requirements
5. **Error Handling**: Implement retry logic for production use cases

## Pricing & Payment Integration

Synthetica includes a complete Stripe payment integration with tiered pricing:

### Pricing Tiers

| Tier | Price | Conversations | Features |
|------|-------|---------------|----------|
| **Free** | $0 | 100 | All features, API access |
| **Starter** | $99 | 1,000 | All features, priority support |
| **Growth** | $450 | 10,000 | All features, priority support, custom integrations |

### Getting Started

1. Visit `/pricing` to create a free account
2. Receive an API key instantly
3. Generate up to 100 conversations for free
4. Upgrade when you need more capacity

### For Developers

See [PAYMENT_INTEGRATION.md](PAYMENT_INTEGRATION.md) for complete documentation on:
- Setting up Stripe integration
- API key management
- Usage tracking and rate limiting
- Webhook configuration
- Testing the payment flow

Quick example:
```bash
# Create a free account
curl -X POST http://localhost:8000/api/users

# Check your usage
curl -H "X-API-Key: sk-synth-..." http://localhost:8000/api/usage

# Generate conversations
curl -X POST http://localhost:8000/api/generate \
  -H "X-API-Key: sk-synth-..." \
  -H "Content-Type: application/json" \
  -d '{"industry":"tech","topics":["bug"],"count":1}'
```

## Requirements

- Python 3.8+
- anthropic >= 0.39.0
- pydantic >= 2.0.0
- fastapi >= 0.104.0 (for web server)
- uvicorn >= 0.24.0 (for web server)
- jinja2 >= 3.1.0 (for web UI)
- python-multipart >= 0.0.6 (for file uploads)
- stripe >= 7.0.0 (for payments)

## License

MIT License - feel free to use this in your projects!

## Contributing

Contributions are welcome! Areas for improvement:

- Additional conversation types (sales, technical support, etc.)
- More sophisticated quality metrics
- Support for multi-language conversations
- Integration with other LLM providers
- Performance optimizations

## Support

For issues, questions, or suggestions, please open an issue on the repository.
