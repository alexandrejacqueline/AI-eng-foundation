from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Représente un bloc de texte avec ses métadonnées de position."""

    text: str = Field(..., description="Contenu textuel du chunk")
    chunk_id: int = Field(
        ..., ge=0, description="Index séquentiel du chunk (0, 1, 2...)"
    )
    start_char: int = Field(
        ...,
        ge=0,
        description="Position du caractère de début dans le document source",
    )
    end_char: int = Field(
        ...,
        ge=0,
        description="Position du caractère de fin dans le document source",
    )


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Découpe un texte en blocs avec chevauchement (sliding window).

    Args:
        text: Le texte complet à découper.
        chunk_size: La taille maximale d'un chunk (en nombre de caractères).
        chunk_overlap: Le nombre de caractères de chevauchement entre deux
          chunks successifs.

    Returns:
        Une liste d'objets Chunk contenant le texte découpé et ses métadonnées.

    Raises:
        ValueError: Si chunk_size <= 0 ou si chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size doit être strictement positif, reçu: {chunk_size}"
        )
    if chunk_overlap < 0:
        raise ValueError(
            f"chunk_overlap doit être positif ou nul, reçu: {chunk_overlap}"
        )
    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap doit être strictement inférieur à chunk_size "
            f"(reçu: chunk_overlap={chunk_overlap}, chunk_size={chunk_size})"
        )

    text_length = len(text)
    if text_length == 0:
        return []

    if text_length <= chunk_size:
        return [Chunk(text=text, chunk_id=0, start_char=0, end_char=text_length)]

    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap
    start = 0
    chunk_id = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_str = text[start:end]

        chunks.append(
            Chunk(
                text=chunk_str,
                chunk_id=chunk_id,
                start_char=start,
                end_char=end,
            )
        )

        if end == text_length:
            break

        start += step
        chunk_id += 1

    return chunks
