"""
Output formatters for exporting conversations to various formats.
"""
import csv
import json
import logging
from pathlib import Path
from typing import List

from synthetica.schemas.conversation import Conversation

logger = logging.getLogger(__name__)


class ConversationFormatter:
    """Formats and exports conversations to various file formats."""

    @staticmethod
    def to_json(conversations: List[Conversation], output_path: str, indent: int = 2) -> None:
        """
        Export conversations to JSON format.

        Args:
            conversations: List of conversations to export
            output_path: Path to output file
            indent: JSON indentation level
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            data = [conv.model_dump(mode='json') for conv in conversations]

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=str, ensure_ascii=False)

            logger.info(f"Exported {len(conversations)} conversations to JSON: {output_path}")

        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            raise

    @staticmethod
    def to_jsonl(conversations: List[Conversation], output_path: str) -> None:
        """
        Export conversations to JSONL (JSON Lines) format.

        Args:
            conversations: List of conversations to export
            output_path: Path to output file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                for conv in conversations:
                    data = conv.model_dump(mode='json')
                    f.write(json.dumps(data, default=str, ensure_ascii=False) + '\n')

            logger.info(f"Exported {len(conversations)} conversations to JSONL: {output_path}")

        except Exception as e:
            logger.error(f"Failed to export to JSONL: {e}")
            raise

    @staticmethod
    def to_csv(conversations: List[Conversation], output_path: str) -> None:
        """
        Export conversations to CSV format.

        Each row represents a complete conversation with flattened structure.

        Args:
            conversations: List of conversations to export
            output_path: Path to output file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow([
                    'conversation_id',
                    'industry',
                    'topic',
                    'tone',
                    'message_count',
                    'quality_score',
                    'conversation_text',
                    'generated_at'
                ])

                # Write conversations
                for conv in conversations:
                    # Flatten conversation into text
                    conversation_text = ConversationFormatter._format_conversation_text(conv)

                    writer.writerow([
                        conv.id,
                        conv.metadata.industry,
                        conv.metadata.topic,
                        conv.metadata.tone,
                        conv.metadata.message_count,
                        conv.quality_score if conv.quality_score is not None else '',
                        conversation_text,
                        conv.metadata.generated_at.isoformat()
                    ])

            logger.info(f"Exported {len(conversations)} conversations to CSV: {output_path}")

        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise

    @staticmethod
    def to_csv_detailed(conversations: List[Conversation], output_path: str) -> None:
        """
        Export conversations to detailed CSV format.

        Each row represents a single message.

        Args:
            conversations: List of conversations to export
            output_path: Path to output file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow([
                    'conversation_id',
                    'message_index',
                    'role',
                    'content',
                    'timestamp',
                    'industry',
                    'topic',
                    'tone',
                    'quality_score'
                ])

                # Write messages
                for conv in conversations:
                    for idx, msg in enumerate(conv.messages):
                        writer.writerow([
                            conv.id,
                            idx,
                            msg.role,
                            msg.content,
                            msg.timestamp.isoformat(),
                            conv.metadata.industry,
                            conv.metadata.topic,
                            conv.metadata.tone,
                            conv.quality_score if conv.quality_score is not None else ''
                        ])

            logger.info(f"Exported {len(conversations)} conversations to detailed CSV: {output_path}")

        except Exception as e:
            logger.error(f"Failed to export to detailed CSV: {e}")
            raise

    @staticmethod
    def _format_conversation_text(conversation: Conversation) -> str:
        """Format conversation as readable text."""
        lines = []
        for msg in conversation.messages:
            role = msg.role.upper()
            lines.append(f"{role}: {msg.content}")
        return " | ".join(lines)

    @staticmethod
    def save_all_formats(
        conversations: List[Conversation],
        base_path: str,
        name: str = "conversations"
    ) -> None:
        """
        Save conversations in all supported formats.

        Args:
            conversations: List of conversations to export
            base_path: Base directory for output files
            name: Base name for output files
        """
        base_dir = Path(base_path)
        base_dir.mkdir(parents=True, exist_ok=True)

        ConversationFormatter.to_json(
            conversations,
            str(base_dir / f"{name}.json")
        )
        ConversationFormatter.to_jsonl(
            conversations,
            str(base_dir / f"{name}.jsonl")
        )
        ConversationFormatter.to_csv(
            conversations,
            str(base_dir / f"{name}.csv")
        )
        ConversationFormatter.to_csv_detailed(
            conversations,
            str(base_dir / f"{name}_detailed.csv")
        )

        logger.info(f"Exported conversations to all formats in {base_path}")
