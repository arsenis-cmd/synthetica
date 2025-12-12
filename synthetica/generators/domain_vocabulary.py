"""
Domain-specific vocabulary and context hints for conversation generation.
"""
from typing import Dict, List, Optional


class DomainVocabulary:
    """Provides domain-specific vocabulary and context hints."""

    DOMAIN_CONFIG = {
        "customer_support": {
            "description": "Customer service and technical support interactions",
            "default_roles": ["customer", "agent"],
            "role_descriptions": {
                "customer": "A person seeking help or information about a product or service",
                "agent": "A support representative helping resolve customer issues"
            },
            "common_vocabulary": [
                "order", "shipping", "tracking", "refund", "return", "exchange",
                "account", "password", "login", "issue", "problem", "error",
                "product", "service", "billing", "payment", "subscription"
            ],
            "typical_scenarios": [
                "order status inquiry",
                "technical troubleshooting",
                "billing question",
                "product return or exchange",
                "account access issue"
            ],
            "tone_guidance": "Professional and helpful, with empathy for customer frustrations"
        },

        "healthcare": {
            "description": "Medical consultations and healthcare interactions",
            "default_roles": ["patient", "doctor"],
            "role_descriptions": {
                "patient": "A person seeking medical advice or discussing health concerns",
                "doctor": "A healthcare provider offering medical guidance and treatment"
            },
            "common_vocabulary": [
                "symptoms", "diagnosis", "treatment", "prescription", "medication",
                "appointment", "test results", "pain", "condition", "fever",
                "allergies", "history", "vitals", "lab work", "follow-up"
            ],
            "typical_scenarios": [
                "symptom discussion and diagnosis",
                "medication consultation",
                "test result review",
                "treatment plan discussion",
                "follow-up appointment scheduling"
            ],
            "tone_guidance": "Professional and empathetic, reassuring but medically accurate"
        },

        "sales": {
            "description": "Sales conversations and product consultations",
            "default_roles": ["prospect", "sales_rep"],
            "role_descriptions": {
                "prospect": "A potential customer interested in learning about products or services",
                "sales_rep": "A sales representative presenting solutions and closing deals"
            },
            "common_vocabulary": [
                "pricing", "demo", "trial", "contract", "implementation", "ROI",
                "features", "benefits", "solution", "package", "discount",
                "timeline", "onboarding", "integration", "license", "upgrade"
            ],
            "typical_scenarios": [
                "product demo request",
                "pricing discussion",
                "contract negotiation",
                "feature comparison",
                "implementation planning"
            ],
            "tone_guidance": "Professional and persuasive, focused on value and solutions"
        },

        "education": {
            "description": "Educational interactions and academic discussions",
            "default_roles": ["student", "teacher"],
            "role_descriptions": {
                "student": "A learner seeking knowledge or clarification on academic topics",
                "teacher": "An educator providing instruction and academic guidance"
            },
            "common_vocabulary": [
                "assignment", "grade", "concept", "deadline", "exam", "quiz",
                "homework", "project", "lecture", "notes", "study", "understand",
                "explanation", "example", "practice", "feedback", "improvement"
            ],
            "typical_scenarios": [
                "homework help and concept clarification",
                "grade discussion",
                "assignment extension request",
                "exam preparation guidance",
                "project feedback session"
            ],
            "tone_guidance": "Educational and encouraging, patient with learning process"
        },

        "legal": {
            "description": "Legal consultations and attorney-client interactions",
            "default_roles": ["client", "lawyer"],
            "role_descriptions": {
                "client": "A person seeking legal advice or representation",
                "lawyer": "A legal professional providing counsel and representation"
            },
            "common_vocabulary": [
                "case", "counsel", "liability", "contract", "agreement", "clause",
                "lawsuit", "settlement", "court", "filing", "evidence", "rights",
                "representation", "consultation", "legal", "attorney", "proceeding"
            ],
            "typical_scenarios": [
                "initial legal consultation",
                "contract review and discussion",
                "case strategy planning",
                "settlement negotiation",
                "legal document preparation"
            ],
            "tone_guidance": "Professional and precise, maintaining client confidentiality"
        },

        "recruiting": {
            "description": "Job interviews and recruitment conversations",
            "default_roles": ["candidate", "recruiter"],
            "role_descriptions": {
                "candidate": "A job seeker being evaluated for a position",
                "recruiter": "A hiring professional assessing candidate fit"
            },
            "common_vocabulary": [
                "role", "experience", "salary", "benefits", "culture fit",
                "skills", "qualifications", "interview", "position", "team",
                "responsibilities", "background", "references", "offer", "start date"
            ],
            "typical_scenarios": [
                "initial phone screen",
                "technical interview",
                "salary negotiation",
                "culture fit assessment",
                "offer discussion"
            ],
            "tone_guidance": "Professional and evaluative, balancing assessment with engagement"
        },

        "financial_services": {
            "description": "Banking, investment, and financial advisory interactions",
            "default_roles": ["client", "advisor"],
            "role_descriptions": {
                "client": "A person seeking financial advice or banking services",
                "advisor": "A financial professional providing guidance and services"
            },
            "common_vocabulary": [
                "account", "balance", "transaction", "investment", "portfolio",
                "interest rate", "loan", "credit", "savings", "retirement",
                "fund", "risk", "return", "fee", "statement", "transfer"
            ],
            "typical_scenarios": [
                "account inquiry",
                "investment consultation",
                "loan application discussion",
                "portfolio review",
                "retirement planning"
            ],
            "tone_guidance": "Professional and trustworthy, financially prudent and clear"
        },

        "real_estate": {
            "description": "Property buying, selling, and rental interactions",
            "default_roles": ["client", "agent"],
            "role_descriptions": {
                "client": "A person looking to buy, sell, or rent property",
                "agent": "A real estate professional facilitating property transactions"
            },
            "common_vocabulary": [
                "property", "listing", "showing", "offer", "inspection", "closing",
                "mortgage", "down payment", "lease", "rent", "deposit", "square feet",
                "neighborhood", "amenities", "market", "price", "appraisal"
            ],
            "typical_scenarios": [
                "property search and showing",
                "offer negotiation",
                "inspection discussion",
                "lease agreement review",
                "closing process walkthrough"
            ],
            "tone_guidance": "Professional and knowledgeable, helping navigate complex transactions"
        }
    }

    @classmethod
    def get_domain_config(cls, domain: str) -> Optional[Dict]:
        """Get configuration for a specific domain."""
        return cls.DOMAIN_CONFIG.get(domain)

    @classmethod
    def get_vocabulary_hints(cls, domain: str) -> List[str]:
        """Get domain-specific vocabulary hints."""
        config = cls.get_domain_config(domain)
        return config["common_vocabulary"] if config else []

    @classmethod
    def get_default_roles(cls, domain: str) -> tuple:
        """Get default role names for a domain."""
        config = cls.get_domain_config(domain)
        if config:
            roles = config["default_roles"]
            return (roles[0], roles[1])
        return ("role_1", "role_2")

    @classmethod
    def get_role_description(cls, domain: str, role: str) -> str:
        """Get description for a specific role in a domain."""
        config = cls.get_domain_config(domain)
        if config and "role_descriptions" in config:
            return config["role_descriptions"].get(role, f"A {role} in {domain}")
        return f"A {role}"

    @classmethod
    def get_typical_scenarios(cls, domain: str) -> List[str]:
        """Get typical scenarios for a domain."""
        config = cls.get_domain_config(domain)
        return config["typical_scenarios"] if config else []

    @classmethod
    def get_tone_guidance(cls, domain: str) -> str:
        """Get tone guidance for a domain."""
        config = cls.get_domain_config(domain)
        return config["tone_guidance"] if config else "Professional and appropriate for the context"

    @classmethod
    def get_supported_domains(cls) -> List[str]:
        """Get list of all supported domains."""
        return list(cls.DOMAIN_CONFIG.keys())

    @classmethod
    def build_context_hint(cls, domain: str, role_1: str, role_2: str, scenario: Optional[str] = None) -> str:
        """
        Build a context hint string for the conversation generator.

        Args:
            domain: The domain of conversation
            role_1: First role name
            role_2: Second role name
            scenario: Optional specific scenario

        Returns:
            Context hint string to guide generation
        """
        config = cls.get_domain_config(domain)

        if not config:
            return f"Conversation between {role_1} and {role_2}"

        hint = f"""Domain: {config['description']}
Roles: {role_1} ({cls.get_role_description(domain, role_1)}) and {role_2} ({cls.get_role_description(domain, role_2)})
Tone: {config['tone_guidance']}"""

        if scenario:
            hint += f"\nScenario: {scenario}"
        else:
            # Provide example scenarios
            scenarios = config['typical_scenarios'][:3]
            hint += f"\nTypical scenarios: {', '.join(scenarios)}"

        # Add vocabulary hints
        vocab = config['common_vocabulary'][:10]
        hint += f"\nCommon vocabulary: {', '.join(vocab)}"

        return hint
