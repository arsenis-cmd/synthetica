"""
Product description and review generator for e-commerce.
"""
import logging
import uuid
from typing import List, Dict, Any
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ProductGenerator:
    """Generates synthetic product descriptions and reviews."""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        """
        Initialize the product generator.

        Args:
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate_product(
        self,
        category: str,
        attributes: Dict[str, Any],
        include_reviews: bool = True,
        review_count: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a synthetic product with description and reviews.

        Args:
            category: Product category (e.g., "electronics", "clothing")
            attributes: Product attributes (price_range, features, etc.)
            include_reviews: Whether to generate customer reviews
            review_count: Number of reviews to generate

        Returns:
            Dictionary containing product data
        """
        try:
            product_id = f"prod_{uuid.uuid4().hex[:12]}"

            # Generate product description
            description_prompt = self._create_description_prompt(category, attributes)
            description = self._generate_text(description_prompt)

            product_data = {
                "id": product_id,
                "category": category,
                "description": description,
                "attributes": attributes,
                "reviews": []
            }

            # Generate reviews if requested
            if include_reviews:
                product_data["reviews"] = self._generate_reviews(
                    category,
                    description,
                    review_count
                )

            logger.info(f"Successfully generated product {product_id}")
            return product_data

        except Exception as e:
            logger.error(f"Failed to generate product: {e}")
            raise

    def _create_description_prompt(self, category: str, attributes: Dict[str, Any]) -> str:
        """Create prompt for product description generation."""
        prompt = f"""Generate a realistic, detailed product description for an e-commerce website.

Category: {category}
Attributes: {', '.join(f'{k}: {v}' for k, v in attributes.items())}

Create a compelling product description that includes:
1. Product name/title
2. Key features and benefits
3. Specifications (where relevant)
4. Use cases
5. What's included

The description should be professional, engaging, and optimized for e-commerce.
Format your response as JSON with fields: title, short_description, features (array), specifications (object), whats_included (array).
"""
        return prompt

    def _generate_reviews(
        self,
        category: str,
        product_description: str,
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate synthetic customer reviews."""
        reviews = []

        review_prompt = f"""Generate {count} realistic customer reviews for this product.

Category: {category}
Product: {product_description[:200]}...

Create diverse reviews with:
- Mix of ratings (1-5 stars, weighted toward positive)
- Different review lengths
- Varied customer perspectives
- Realistic pros and cons
- Natural language

Format as JSON array with fields: rating, title, comment, helpful_count, verified_purchase."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.8,
                messages=[{"role": "user", "content": review_prompt}]
            )

            # Parse reviews from response
            # In production, add proper JSON parsing and validation
            reviews_text = response.content[0].text
            logger.info(f"Generated {count} reviews")

            # Placeholder: In production, parse JSON and return structured data
            return [{"raw_response": reviews_text}]

        except Exception as e:
            logger.error(f"Failed to generate reviews: {e}")
            return []

    def _generate_text(self, prompt: str) -> str:
        """Generate text using Claude API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise

    def generate_batch(
        self,
        category: str,
        count: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple products in batch.

        Args:
            category: Product category
            count: Number of products to generate
            **kwargs: Additional arguments passed to generate_product

        Returns:
            List of product dictionaries
        """
        products = []

        for i in range(count):
            try:
                logger.info(f"Generating product {i+1}/{count}")

                # Generate varied attributes for each product
                attributes = self._generate_attributes(category)
                product = self.generate_product(category, attributes, **kwargs)
                products.append(product)

            except Exception as e:
                logger.error(f"Failed to generate product {i+1}: {e}")
                continue

        return products

    def _generate_attributes(self, category: str) -> Dict[str, Any]:
        """Generate varied product attributes based on category."""
        # Placeholder: In production, use more sophisticated attribute generation
        base_attributes = {
            "electronics": {
                "price_range": "$100-$500",
                "condition": "new",
                "warranty": "1 year"
            },
            "clothing": {
                "price_range": "$30-$100",
                "material": "cotton blend",
                "sizes": ["S", "M", "L", "XL"]
            },
            "home": {
                "price_range": "$50-$200",
                "dimensions": "varies",
                "assembly_required": False
            }
        }

        return base_attributes.get(category, {"price_range": "$50-$150"})
