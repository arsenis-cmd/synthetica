"""
FastAPI web server for Synthetica conversation generation.
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel, Field

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

# Initialize FastAPI app
app = FastAPI(
    title="Synthetica API",
    description="Synthetic customer support conversation generation platform",
    version="1.0.0"
)

# Setup templates
template_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# Request/Response Models
class GenerateRequest(BaseModel):
    """Request model for conversation generation."""
    industry: str = Field(..., description="Industry context (e.g., technology, healthcare)")
    topics: List[str] = Field(..., min_length=1, description="List of conversation topics")
    count: int = Field(default=1, ge=1, le=50, description="Number of conversations to generate")
    customer_tone: str = Field(default="professional", description="Tone for customer messages")
    agent_style: str = Field(default="professional", description="Style for agent responses")
    message_count: int = Field(default=8, ge=4, le=20, description="Messages per conversation")


class ConversationResponse(BaseModel):
    """Response model for a single conversation."""
    id: str
    industry: str
    topic: str
    tone: str
    message_count: int
    quality_score: Optional[float]
    messages: List[dict]


class GenerateResponse(BaseModel):
    """Response model for generation endpoint."""
    success: bool
    count: int
    conversations: List[ConversationResponse]
    average_quality: Optional[float]
    output_files: dict


class DatasetInfo(BaseModel):
    """Information about a dataset file."""
    filename: str
    size_bytes: int
    created_at: str
    format: str


class DatasetsResponse(BaseModel):
    """Response model for datasets listing."""
    datasets: List[DatasetInfo]
    total_count: int


# Helper functions
def get_api_key() -> str:
    """Get Anthropic API key from environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY environment variable not set"
        )
    return api_key


def get_file_info(file_path: Path) -> DatasetInfo:
    """Get information about a dataset file."""
    stat = file_path.stat()
    format_map = {
        '.json': 'JSON',
        '.jsonl': 'JSONL',
        '.csv': 'CSV'
    }

    return DatasetInfo(
        filename=file_path.name,
        size_bytes=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        format=format_map.get(file_path.suffix, 'Unknown')
    )


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_conversations(request: GenerateRequest):
    """
    Generate synthetic customer support conversations.

    Args:
        request: Generation configuration

    Returns:
        Generated conversations with quality scores
    """
    try:
        logger.info(f"Generating {request.count} conversations for industry: {request.industry}")

        # Get API key
        api_key = get_api_key()

        # Initialize generator and scorer
        generator = ConversationGenerator(api_key=api_key)
        scorer = QualityScorer(api_key=api_key)

        # Create config - use the customer_tone as the tone for the conversation
        config = ConversationConfig(
            industry=request.industry,
            topics=request.topics,
            tone=request.customer_tone,  # Use customer_tone as the overall tone
            message_count=request.message_count
        )

        # Generate conversations
        conversations = []
        for i in range(request.count):
            try:
                logger.info(f"Generating conversation {i+1}/{request.count}")
                conv = generator.generate(config)

                # Score the conversation
                quality = scorer.score(conv, use_ai=False)
                conv.quality_score = quality.overall

                conversations.append(conv)
            except Exception as e:
                logger.error(f"Failed to generate conversation {i+1}: {e}")
                continue

        if not conversations:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate any conversations"
            )

        # Calculate average quality
        avg_quality = sum(c.quality_score for c in conversations if c.quality_score) / len(conversations)

        # Save to output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{request.industry}_{timestamp}"

        output_files = {
            "json": str(OUTPUT_DIR / f"{base_name}.json"),
            "jsonl": str(OUTPUT_DIR / f"{base_name}.jsonl"),
            "csv": str(OUTPUT_DIR / f"{base_name}.csv"),
            "csv_detailed": str(OUTPUT_DIR / f"{base_name}_detailed.csv")
        }

        # Export to all formats
        ConversationFormatter.save_all_formats(
            conversations,
            str(OUTPUT_DIR),
            name=base_name
        )

        # Convert to response format
        conversation_responses = [
            ConversationResponse(
                id=conv.id,
                industry=conv.metadata.industry,
                topic=conv.metadata.topic,
                tone=conv.metadata.tone,
                message_count=len(conv.messages),
                quality_score=conv.quality_score,
                messages=[
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat()
                    }
                    for msg in conv.messages
                ]
            )
            for conv in conversations
        ]

        logger.info(f"Successfully generated {len(conversations)} conversations")

        return GenerateResponse(
            success=True,
            count=len(conversations),
            conversations=conversation_responses,
            average_quality=round(avg_quality, 2),
            output_files=output_files
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets", response_model=DatasetsResponse)
async def list_datasets():
    """
    List all generated datasets in the output folder.

    Returns:
        List of dataset files with metadata
    """
    try:
        datasets = []

        # List all files in output directory
        if OUTPUT_DIR.exists():
            for file_path in OUTPUT_DIR.iterdir():
                if file_path.is_file() and file_path.suffix in ['.json', '.jsonl', '.csv']:
                    datasets.append(get_file_info(file_path))

        # Sort by created_at descending
        datasets.sort(key=lambda x: x.created_at, reverse=True)

        return DatasetsResponse(
            datasets=datasets,
            total_count=len(datasets)
        )

    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets/{filename}")
async def download_dataset(filename: str):
    """
    Download a specific dataset file.

    Args:
        filename: Name of the file to download

    Returns:
        File download response
    """
    try:
        file_path = OUTPUT_DIR / filename

        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        # Security check - ensure file is in output directory
        if not str(file_path.resolve()).startswith(str(OUTPUT_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_key_set": bool(os.getenv("ANTHROPIC_API_KEY"))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
