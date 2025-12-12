"""
FastAPI web server for Synthetica conversation generation with Stripe integration.
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import stripe
from fastapi import FastAPI, HTTPException, Header, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from synthetica.generators.conversation import ConversationGenerator
from synthetica.generators.product import ProductGenerator
from synthetica.generators.diversity import DiversityEngine
from synthetica.quality.scorer import QualityScorer
from synthetica.schemas.conversation import ConversationConfig
from synthetica.output.formatters import ConversationFormatter
from synthetica.api.database import DatabaseManager, SubscriptionTier, TIER_PRICES, TIER_LIMITS
from synthetica.analysis.seed_analyzer import SeedAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Synthetica API",
    description="Synthetic customer support conversation generation platform with usage tracking",
    version="2.0.0"
)

# Setup templates
template_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize database
db = DatabaseManager()

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")


# Request/Response Models
class DiversityConfig(BaseModel):
    """Configuration for diversity engine."""
    enabled: bool = Field(default=False, description="Enable diversity engine")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold for anti-repetition")
    min_diversity_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum acceptable diversity score")
    seed_examples: Optional[List[str]] = Field(None, description="Seed conversation examples for style injection (5-50)")


class GenerateRequest(BaseModel):
    """Request model for conversation generation."""
    industry: str = Field(..., description="Industry context")
    topics: List[str] = Field(..., min_length=1, description="List of conversation topics")
    count: int = Field(default=1, ge=1, le=50, description="Number of conversations to generate")
    customer_tone: str = Field(default="professional")
    agent_style: str = Field(default="professional")
    message_count: int = Field(default=8, ge=4, le=20)
    diversity_config: Optional[DiversityConfig] = Field(None, description="Optional diversity engine configuration")
    use_seeds: bool = Field(default=False, description="Use uploaded seed conversations for style-based generation")


class QualityMetrics(BaseModel):
    """Detailed quality metrics."""
    coherence: float
    diversity: float
    naturalness: float
    overall: float


class ConversationResponse(BaseModel):
    """Response model for a single conversation."""
    id: str
    industry: str
    topic: str
    tone: str
    message_count: int
    quality_score: Optional[float]
    quality_metrics: Optional[QualityMetrics]
    messages: List[dict]


class DiversityMetrics(BaseModel):
    """Diversity metrics for batch of conversations."""
    overall: float
    vocabulary_diversity: float
    structure_diversity: float
    persona_consistency: float
    meets_threshold: bool
    min_threshold: float


class GenerateResponse(BaseModel):
    """Response model for generation endpoint."""
    success: bool
    count: int
    conversations: List[ConversationResponse]
    average_quality: Optional[float]
    diversity_metrics: Optional[DiversityMetrics] = None
    output_files: dict
    usage_info: dict


class CreateUserRequest(BaseModel):
    """Request to create a new user."""
    email: Optional[str] = None


class CreateUserResponse(BaseModel):
    """Response with new API key."""
    api_key: str
    subscription_tier: str
    usage_limit: int


class CheckoutRequest(BaseModel):
    """Request to create checkout session."""
    tier: str = Field(..., description="Subscription tier (starter or growth)")
    success_url: str = Field(..., description="URL to redirect on success")
    cancel_url: str = Field(..., description="URL to redirect on cancel")


class CheckoutResponse(BaseModel):
    """Response with checkout session URL."""
    session_id: str
    checkout_url: str


class UsageResponse(BaseModel):
    """Response with usage information."""
    tier: str
    usage: int
    limit: int
    remaining: int
    percentage_used: float


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


class GenerateProductRequest(BaseModel):
    """Request model for product generation."""
    category: str = Field(..., description="Product category (electronics, clothing, home, etc.)")
    count: int = Field(default=1, ge=1, le=20, description="Number of products to generate")
    include_reviews: bool = Field(default=True, description="Include customer reviews")
    review_count: int = Field(default=5, ge=1, le=10, description="Number of reviews per product")


class ProductResponse(BaseModel):
    """Response model for a single product."""
    id: str
    category: str
    description: str
    attributes: dict
    reviews: List[dict]


class GenerateProductResponse(BaseModel):
    """Response model for product generation endpoint."""
    success: bool
    count: int
    products: List[ProductResponse]
    output_file: Optional[str]
    usage_info: dict


class UploadSeedsRequest(BaseModel):
    """Request model for uploading seed conversations."""
    conversations: List[dict] = Field(
        ...,
        min_length=10,
        max_length=50,
        description="List of 10-50 seed conversations with messages"
    )
    generate_personas: bool = Field(
        default=True,
        description="Automatically generate personas from seeds"
    )
    persona_count: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Number of personas to generate (3-10)"
    )


class UploadSeedsResponse(BaseModel):
    """Response model for seed upload endpoint."""
    success: bool
    seed_count: int
    analysis: dict
    personas: Optional[List[dict]] = None
    message: str


# Helper functions
def get_api_key_from_header(x_api_key: Optional[str] = Header(None)) -> str:
    """Extract and validate API key from header."""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include X-API-Key header."
        )
    return x_api_key


def get_anthropic_api_key() -> str:
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
    format_map = {'.json': 'JSON', '.jsonl': 'JSONL', '.csv': 'CSV'}

    return DatasetInfo(
        filename=file_path.name,
        size_bytes=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        format=format_map.get(file_path.suffix, 'Unknown')
    )


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def landing(request: FastAPIRequest):
    """Serve the landing page."""
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/generate", response_class=HTMLResponse)
async def generate_page(request: FastAPIRequest):
    """Serve the conversation generator interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/pricing", response_class=HTMLResponse)
async def pricing(request: FastAPIRequest):
    """Serve the pricing page."""
    return templates.TemplateResponse("pricing.html", {"request": request})


@app.get("/docs", response_class=HTMLResponse)
async def docs_page(request: FastAPIRequest):
    """Serve the API documentation page."""
    return templates.TemplateResponse("docs.html", {"request": request})


@app.get("/marketplace", response_class=HTMLResponse)
async def marketplace(request: FastAPIRequest):
    """Serve the dataset marketplace page."""
    return templates.TemplateResponse("marketplace.html", {"request": request})


@app.get("/generate/products", response_class=HTMLResponse)
async def generate_products_page(request: FastAPIRequest):
    """Serve the product generator interface."""
    return templates.TemplateResponse("products.html", {"request": request})


@app.post("/api/users", response_model=CreateUserResponse)
async def create_user(request: CreateUserRequest):
    """
    Create a new user with a free tier API key.

    Returns:
        New API key and subscription info
    """
    try:
        api_key, user = db.create_user(email=request.email)

        return CreateUserResponse(
            api_key=api_key,
            subscription_tier=user['subscription_tier'],
            usage_limit=TIER_LIMITS[user['subscription_tier']]
        )
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/usage", response_model=UsageResponse)
async def get_usage(x_api_key: str = Header(..., alias="X-API-Key")):
    """
    Get current usage information for the API key.

    Headers:
        X-API-Key: Your Synthetica API key
    """
    within_limit, usage_info = db.check_usage_limit(x_api_key)

    if "error" in usage_info:
        raise HTTPException(status_code=401, detail=usage_info["error"])

    return UsageResponse(
        tier=usage_info["tier"],
        usage=usage_info["usage"],
        limit=usage_info["limit"],
        remaining=usage_info["remaining"],
        percentage_used=round((usage_info["usage"] / usage_info["limit"] * 100)
                             if usage_info["limit"] > 0 else 0, 1)
    )


@app.post("/api/upload-seeds", response_model=UploadSeedsResponse)
async def upload_seed_conversations(
    request: UploadSeedsRequest,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """
    Upload seed conversations for style-based generation.

    Accepts 10-50 example conversations, analyzes their style patterns,
    and optionally generates personas that match the detected style.

    Headers:
        X-API-Key: Your Synthetica API key

    The seed conversations will be used when generating with use_seeds=true.
    """
    try:
        # Validate API key
        user = db.get_user_by_api_key(x_api_key)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")

        logger.info(f"Uploading {len(request.conversations)} seed conversations")

        # Validate conversations structure
        for i, conv in enumerate(request.conversations):
            if 'messages' not in conv:
                raise HTTPException(
                    status_code=400,
                    detail=f"Conversation {i} missing 'messages' field"
                )

            messages = conv['messages']
            if not isinstance(messages, list) or len(messages) < 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Conversation {i} must have at least 2 messages"
                )

            # Check message structure
            for j, msg in enumerate(messages):
                if 'role' not in msg or 'content' not in msg:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Message {j} in conversation {i} missing 'role' or 'content'"
                    )

        # Store seed conversations
        success = db.store_seed_conversations(x_api_key, request.conversations)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store seed conversations")

        # Analyze seeds
        analyzer = SeedAnalyzer()
        analysis = analyzer.analyze_seeds(request.conversations)

        # Store analysis
        db.store_seed_analysis(x_api_key, analysis)

        # Generate personas if requested
        personas = None
        if request.generate_personas:
            logger.info(f"Generating {request.persona_count} personas from seed analysis")
            personas = analyzer.generate_personas_from_analysis(
                analysis,
                count=request.persona_count
            )

            # Store personas
            for persona in personas:
                db.store_seed_persona(x_api_key, persona)

        return UploadSeedsResponse(
            success=True,
            seed_count=len(request.conversations),
            analysis=analysis,
            personas=personas,
            message=f"Successfully uploaded {len(request.conversations)} seeds and generated {len(personas) if personas else 0} personas"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading seeds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_conversations(
    request: GenerateRequest,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """
    Generate synthetic customer support conversations.

    Headers:
        X-API-Key: Your Synthetica API key

    Rate Limits:
        - Free: 100 conversations
        - Starter: 1,000 conversations
        - Growth: 10,000 conversations
    """
    try:
        # Check usage limit
        within_limit, usage_info = db.check_usage_limit(x_api_key)

        if not within_limit:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Usage limit exceeded",
                    "tier": usage_info["tier"],
                    "limit": usage_info["limit"],
                    "usage": usage_info["usage"],
                    "upgrade_url": "/pricing"
                }
            )

        # Check if request would exceed limit
        if usage_info["remaining"] < request.count:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Request would exceed usage limit",
                    "remaining": usage_info["remaining"],
                    "requested": request.count,
                    "upgrade_url": "/pricing"
                }
            )

        logger.info(f"Generating {request.count} conversations")

        # Get Anthropic API key
        anthropic_key = get_anthropic_api_key()

        # Handle seed-based generation
        seed_personas = None
        seed_analysis = None
        if request.use_seeds:
            logger.info("Using seed-based generation")

            # Get seed analysis
            seed_analysis = db.get_seed_analysis(x_api_key)
            if not seed_analysis:
                raise HTTPException(
                    status_code=400,
                    detail="No seed conversations found. Upload seeds first using /api/upload-seeds"
                )

            # Get seed personas
            seed_personas = db.get_seed_personas(x_api_key)
            if not seed_personas:
                raise HTTPException(
                    status_code=400,
                    detail="No seed personas found. Upload seeds with generate_personas=true"
                )

            logger.info(f"Found {len(seed_personas)} seed personas to use")

        # Initialize diversity engine if configured
        diversity_engine = None
        if request.diversity_config and request.diversity_config.enabled:
            logger.info("Initializing diversity engine")
            diversity_engine = DiversityEngine(
                similarity_threshold=request.diversity_config.similarity_threshold,
                min_diversity_score=request.diversity_config.min_diversity_score
            )

            # Load seed examples if provided
            if request.diversity_config.seed_examples:
                logger.info(f"Loading {len(request.diversity_config.seed_examples)} seed examples")
                diversity_engine.load_seed_examples(request.diversity_config.seed_examples)

        # Initialize generator and scorer
        generator = ConversationGenerator(
            api_key=anthropic_key,
            diversity_engine=diversity_engine
        )
        scorer = QualityScorer(api_key=anthropic_key)

        # Create config
        config = ConversationConfig(
            industry=request.industry,
            topics=request.topics,
            tone=request.customer_tone,
            message_count=request.message_count
        )

        # Generate conversations
        conversations = []
        quality_details = {}  # Store detailed quality metrics
        conversation_texts = []  # For diversity scoring

        for i in range(request.count):
            try:
                logger.info(f"Generating conversation {i+1}/{request.count}")

                # Use seed persona if available
                current_persona = None
                if seed_personas:
                    # Rotate through personas
                    persona_dict = seed_personas[i % len(seed_personas)]
                    # Convert dict to Persona-like object
                    from synthetica.generators.diversity import Persona
                    current_persona = Persona(**persona_dict)
                    logger.info(f"Using seed persona: {persona_dict.get('id', 'unknown')}")

                # Generate with or without persona
                if current_persona and not diversity_engine:
                    # Create temporary diversity engine just for persona
                    temp_diversity_engine = DiversityEngine()
                    temp_diversity_engine.personas = [current_persona]
                    temp_generator = ConversationGenerator(
                        api_key=anthropic_key,
                        diversity_engine=temp_diversity_engine
                    )
                    conv = temp_generator.generate(config)
                else:
                    conv = generator.generate(config)

                # Add persona_id to metadata if using seeds
                if seed_personas and conv.metadata:
                    persona_dict = seed_personas[i % len(seed_personas)]
                    conv.metadata.persona = persona_dict

                # Check anti-repetition if diversity engine is enabled
                if diversity_engine:
                    # Format conversation as text for diversity checking
                    conv_text = " ".join(m.content for m in conv.messages)

                    # Validate against repetition
                    validation = diversity_engine.validate_conversation(conv_text)

                    if not validation["valid"]:
                        logger.warning(
                            f"Conversation {i+1} flagged as too similar "
                            f"(similarity: {validation['similarity_score']:.2f}), "
                            "regenerating..."
                        )
                        # Skip this conversation and try again (don't increment i)
                        continue

                    # Add to tracking
                    diversity_engine.add_conversation(conv_text)
                    conversation_texts.append(conv_text)

                # Score the conversation
                quality = scorer.score(conv, use_ai=False)
                conv.quality_score = quality.overall
                quality_details[conv.id] = quality  # Store full quality object

                conversations.append(conv)

                # Increment usage count
                db.increment_usage(x_api_key)

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

        # Save to output directory
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

        # Calculate diversity metrics if diversity engine or seed-based generation was used
        diversity_metrics = None
        if (diversity_engine or request.use_seeds) and len(conversations) > 1:
            logger.info("Calculating diversity metrics for batch")

            # Collect conversation texts if not already done
            if not conversation_texts:
                conversation_texts = [" ".join(m.content for m in conv.messages) for conv in conversations]

            # Convert conversations to format needed for diversity scoring
            conversation_objects = [
                {
                    "id": conv.id,
                    "metadata": {
                        "persona": conv.metadata.persona if hasattr(conv.metadata, 'persona') else None
                    }
                }
                for conv in conversations
            ]

            # Use existing diversity engine or create one for scoring
            scoring_engine = diversity_engine if diversity_engine else DiversityEngine(
                similarity_threshold=0.8  # Flag if similarity > 80%
            )

            diversity_scores = scoring_engine.calculate_batch_diversity(
                conversation_texts,
                conversation_objects
            )

            # Flag conversations that are too similar
            high_similarity_pairs = []
            for i, conv_text_1 in enumerate(conversation_texts):
                for j, conv_text_2 in enumerate(conversation_texts[i+1:], i+1):
                    similarity = scoring_engine.calculate_text_similarity(conv_text_1, conv_text_2)
                    if similarity > 0.8:
                        high_similarity_pairs.append({
                            "conversation_1": conversations[i].id,
                            "conversation_2": conversations[j].id,
                            "similarity": round(similarity, 3)
                        })

            # Add similarity warnings to diversity scores
            diversity_scores["high_similarity_pairs"] = high_similarity_pairs
            diversity_scores["has_high_similarity"] = len(high_similarity_pairs) > 0

            diversity_metrics = DiversityMetrics(
                overall=diversity_scores["overall"],
                vocabulary_diversity=diversity_scores["vocabulary_diversity"],
                structure_diversity=diversity_scores["structure_diversity"],
                persona_consistency=diversity_scores["persona_consistency"],
                meets_threshold=diversity_scores["meets_threshold"],
                min_threshold=diversity_scores["min_threshold"]
            )

            if not diversity_scores["meets_threshold"]:
                logger.warning(
                    f"Batch diversity score {diversity_scores['overall']:.2f} "
                    f"is below threshold {diversity_scores['min_threshold']:.2f}"
                )

        # Get updated usage info
        _, updated_usage = db.check_usage_limit(x_api_key)

        # Convert to response format
        conversation_responses = [
            ConversationResponse(
                id=conv.id,
                industry=conv.metadata.industry,
                topic=conv.metadata.topic,
                tone=conv.metadata.tone,
                message_count=len(conv.messages),
                quality_score=conv.quality_score,
                quality_metrics=QualityMetrics(
                    coherence=quality_details[conv.id].coherence,
                    diversity=quality_details[conv.id].diversity,
                    naturalness=quality_details[conv.id].naturalness,
                    overall=quality_details[conv.id].overall
                ) if conv.id in quality_details else None,
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

        return GenerateResponse(
            success=True,
            count=len(conversations),
            conversations=conversation_responses,
            average_quality=round(avg_quality, 2),
            diversity_metrics=diversity_metrics,
            output_files=output_files,
            usage_info=updated_usage
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/products", response_model=GenerateProductResponse)
async def generate_products(
    request: GenerateProductRequest,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """
    Generate synthetic product descriptions and reviews.

    Headers:
        X-API-Key: Your Synthetica API key

    Rate Limits:
        - Same as conversation generation
    """
    try:
        # Check usage limit
        within_limit, usage_info = db.check_usage_limit(x_api_key)

        if not within_limit:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Usage limit exceeded",
                    "tier": usage_info["tier"],
                    "limit": usage_info["limit"],
                    "usage": usage_info["usage"],
                    "upgrade_url": "/pricing"
                }
            )

        # Check if request would exceed limit
        if usage_info["remaining"] < request.count:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Request would exceed usage limit",
                    "remaining": usage_info["remaining"],
                    "requested": request.count,
                    "upgrade_url": "/pricing"
                }
            )

        logger.info(f"Generating {request.count} products in category: {request.category}")

        # Get Anthropic API key
        anthropic_key = get_anthropic_api_key()

        # Initialize product generator
        generator = ProductGenerator(api_key=anthropic_key)

        # Generate products
        products = generator.generate_batch(
            category=request.category,
            count=request.count,
            include_reviews=request.include_reviews,
            review_count=request.review_count
        )

        if not products:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate any products"
            )

        # Increment usage for each product generated
        for _ in range(len(products)):
            db.increment_usage(x_api_key)

        # Save to output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{request.category}_products_{timestamp}.json"
        output_path = OUTPUT_DIR / filename

        import json
        with open(output_path, 'w') as f:
            json.dump(products, f, indent=2)

        logger.info(f"Saved {len(products)} products to {output_path}")

        # Get updated usage info
        _, updated_usage = db.check_usage_limit(x_api_key)

        # Convert to response format
        product_responses = [
            ProductResponse(
                id=prod["id"],
                category=prod["category"],
                description=prod["description"],
                attributes=prod["attributes"],
                reviews=prod.get("reviews", [])
            )
            for prod in products
        ]

        return GenerateProductResponse(
            success=True,
            count=len(products),
            products=product_responses,
            output_file=str(output_path),
            usage_info=updated_usage
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating products: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/checkout", response_model=CheckoutResponse)
async def create_checkout_session(
    request: CheckoutRequest,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """
    Create a Stripe checkout session for subscription upgrade.

    Headers:
        X-API-Key: Your Synthetica API key
    """
    try:
        # Validate tier
        if request.tier not in [SubscriptionTier.STARTER, SubscriptionTier.GROWTH]:
            raise HTTPException(
                status_code=400,
                detail="Invalid tier. Choose 'starter' or 'growth'"
            )

        # Get user
        user = db.get_user_by_api_key(x_api_key)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Get price
        amount = TIER_PRICES[request.tier]

        # Create Stripe checkout session
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'unit_amount': amount,
                    'product_data': {
                        'name': f'Synthetica {request.tier.capitalize()} Plan',
                        'description': f'{TIER_LIMITS[request.tier]:,} conversations',
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            client_reference_id=x_api_key,
            metadata={
                'tier': request.tier,
                'api_key': x_api_key
            }
        )

        # Store payment record
        db.create_payment(
            api_key=x_api_key,
            stripe_session_id=session.id,
            amount=amount,
            subscription_tier=request.tier
        )

        return CheckoutResponse(
            session_id=session.id,
            checkout_url=session.url
        )

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating checkout session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/webhooks/stripe")
async def stripe_webhook(request: FastAPIRequest):
    """
    Handle Stripe webhook events.

    This endpoint processes payment confirmations and upgrades subscriptions.
    """
    try:
        payload = await request.body()
        sig_header = request.headers.get('stripe-signature')

        # Verify webhook signature
        if STRIPE_WEBHOOK_SECRET:
            try:
                event = stripe.Webhook.construct_event(
                    payload, sig_header, STRIPE_WEBHOOK_SECRET
                )
            except stripe.error.SignatureVerificationError as e:
                logger.error(f"Invalid signature: {e}")
                raise HTTPException(status_code=400, detail="Invalid signature")
        else:
            # For testing without webhook secret
            event = stripe.Event.construct_from(
                stripe.util.json.loads(payload), stripe.api_key
            )

        # Handle checkout session completed
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            session_id = session['id']
            payment_intent = session.get('payment_intent')

            logger.info(f"Payment completed for session {session_id}")

            # Complete the payment and upgrade user
            db.complete_payment(session_id, payment_intent)

        return JSONResponse(content={"status": "success"})

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets", response_model=DatasetsResponse)
async def list_datasets():
    """List all generated datasets in the output folder."""
    try:
        datasets = []

        if OUTPUT_DIR.exists():
            for file_path in OUTPUT_DIR.iterdir():
                if file_path.is_file() and file_path.suffix in ['.json', '.jsonl', '.csv']:
                    datasets.append(get_file_info(file_path))

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
    """Download a specific dataset file."""
    try:
        file_path = OUTPUT_DIR / filename

        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        # Security check
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
        "anthropic_api_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
        "stripe_configured": bool(stripe.api_key)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
