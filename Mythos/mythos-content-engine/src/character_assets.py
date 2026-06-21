"""Shared Mortal Vengeance character asset registry."""

from __future__ import annotations

import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHARACTER_NAME_DIR = PROJECT_ROOT / "assets" / "character_names"
CHARACTER_PORTRAIT_DIR = PROJECT_ROOT / "assets" / "character_portraits" / "book1"
CHARACTER_PORTRAIT_DIR_BOOK2 = PROJECT_ROOT / "assets" / "character_portraits" / "book2"


CHARACTER_NAME_ASSETS = {
    "Alex Herrera": "alex_herrera.png",
    "Melissa Rocha": "melissa_rocha.png",
    "Mónika Torres": "monika_torres.png",
    "Mario Stinga": "mario_stinga.png",
    "Manuel Freites": "manuel_freites.png",
    "Fernando Pepino": "fernando_pepino.png",
    "Enrique Hartling": "enrique_hartling.png",
    "María García": "maria_garcia.png",
    "Julián Díaz": "julian_diaz.png",
    "Lucía Salgado": "lucia_salgado.png",
    "Marcos": "marcos.png",
    "Profesora Lourdes": "profesora_lourdes.png",
    "Lieutenant Ricardo García": "lieutenant_ricardo_garcia.png",
    "The Grim Cojuelo": "the_grim_cojuelo.png",
    "Doña Silvia": "dona_silvia.png",
    "Padre Ángel": "padre_angel.png",
    "Sister María Gracia": "sister_maria_gracia.png",
    "Padre Ignacio": "padre_ignacio.png",
}


BOOK1_CHARACTER_PORTRAITS = {
    "Alex Herrera": "alex_herrera.png",
    "Melissa Rocha": "melissa_rocha.png",
    "Marcos": "marcos.png",
    "Manuel Freites": "manuel_freites.png",
    "Lieutenant Ricardo García": "lieutenant_ricardo_garcia.png",
    "The Grim Cojuelo": "the_grim_cojuelo.png",
    "Fernando Pepino": "fernando_pepino.png",
    "Enrique Hartling": "enrique_hartling.png",
    "María García": "maria_garcia.png",
    "Mario Stinga": "mario_stinga.png",
    "Mónika Torres": "monika_torres.png",
    "Profesora Lourdes": "profesora_lourdes.png",
}


# Portraits for Mortal Vengeance II: To Reel or Not Too Real? (Book 2). Several
# characters recur from Book 1 (Alex, Mario, Melissa, Mónika) but get a distinct
# Book 2 portrait; the rest are new to Book 2.
BOOK2_CHARACTER_PORTRAITS = {
    "Alex Herrera": "alex_herrera.png",
    "Mario Stinga": "mario_stinga.png",
    "Lucía Salgado": "lucia_salgado.png",
    "Valeria Viccini": "valeria_viccini.png",
    "Doña Silvia": "dona_silvia.png",
    "Camila Álvarez": "camila_alvarez.png",
    "Rafael Montero": "rafael_montero.png",
    "Shane Harper": "shane_harper.png",
    "Melissa Rocha": "melissa_rocha.png",
    "Mónika Torres": "monika_torres.png",
}


_ASSET_ALIASES = {
    "alex": "Alex Herrera",
    "alex herrera": "Alex Herrera",
    "melissa": "Melissa Rocha",
    "melissa rocha": "Melissa Rocha",
    "marcos": "Marcos",
    "manuel": "Manuel Freites",
    "manuel freites": "Manuel Freites",
    "ricardo": "Lieutenant Ricardo García",
    "lieutenant ricardo": "Lieutenant Ricardo García",
    "lietenaunt ricardo": "Lieutenant Ricardo García",
    "lieutenant ricardo garcia": "Lieutenant Ricardo García",
    "lietenaunt ricardo garcia": "Lieutenant Ricardo García",
    "the grim cojuelo": "The Grim Cojuelo",
    "grim cojuelo": "The Grim Cojuelo",
    "fernando": "Fernando Pepino",
    "fernando pepino": "Fernando Pepino",
    "enrique": "Enrique Hartling",
    "enrique hartling": "Enrique Hartling",
    "maria": "María García",
    "maria garcia": "María García",
    "mario": "Mario Stinga",
    "mario stinga": "Mario Stinga",
    "monika": "Mónika Torres",
    "monika torres": "Mónika Torres",
    "monica": "Mónika Torres",
    "monica torres": "Mónika Torres",
    "profesora lourdes": "Profesora Lourdes",
    "lourdes": "Profesora Lourdes",
    # Mortal Vengeance II characters
    "lucia": "Lucía Salgado",
    "lucia salgado": "Lucía Salgado",
    "valeria": "Valeria Viccini",
    "valeria viccini": "Valeria Viccini",
    "camila": "Camila Álvarez",
    "camila alvarez": "Camila Álvarez",
    "rafa": "Rafael Montero",
    "rafael": "Rafael Montero",
    "rafa montero": "Rafael Montero",
    "rafael montero": "Rafael Montero",
    "shane": "Shane Harper",
    "shane harper": "Shane Harper",
    "dona silvia": "Doña Silvia",
    "silvia": "Doña Silvia",
}


def normalize_asset_key(value: str) -> str:
    """Normalize names and filename fragments for asset lookup."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(ascii_text.replace("_", " ").replace("-", " ").lower().split())


def canonical_character_name(character_name) -> str:
    """Return the canonical character name for selected UI values or filename hints."""
    if isinstance(character_name, (list, tuple)):
        character_name = next((item for item in character_name if item), "")
    text = str(character_name or "").strip()
    if not text:
        return ""
    if (
        text in CHARACTER_NAME_ASSETS
        or text in BOOK1_CHARACTER_PORTRAITS
        or text in BOOK2_CHARACTER_PORTRAITS
    ):
        return text
    return _ASSET_ALIASES.get(normalize_asset_key(text), text)


def _is_book2(book) -> bool:
    """True when the selected book/source refers to Mortal Vengeance II."""
    return "mortal vengeance ii" in normalize_asset_key(book)


def resolve_character_name_asset(character_name) -> Path | None:
    """Return the uploaded character-name mark for a selected character."""
    canonical_name = canonical_character_name(character_name)
    filename = CHARACTER_NAME_ASSETS.get(canonical_name)
    if not filename:
        return None
    path = CHARACTER_NAME_DIR / filename
    return path if path.exists() else None


def resolve_character_portrait_asset(character_name, book=None) -> Path | None:
    """Return the portrait for a selected character, when available.

    When ``book`` refers to Mortal Vengeance II, the Book 2 portrait is preferred
    (recurring characters have a distinct Book 2 look); otherwise Book 1 is
    preferred. Either way the other book is used as a fallback, so characters who
    only have a portrait in one book still resolve regardless of the selection.
    """
    canonical_name = canonical_character_name(character_name)
    if _is_book2(book):
        search = (
            (BOOK2_CHARACTER_PORTRAITS, CHARACTER_PORTRAIT_DIR_BOOK2),
            (BOOK1_CHARACTER_PORTRAITS, CHARACTER_PORTRAIT_DIR),
        )
    else:
        search = (
            (BOOK1_CHARACTER_PORTRAITS, CHARACTER_PORTRAIT_DIR),
            (BOOK2_CHARACTER_PORTRAITS, CHARACTER_PORTRAIT_DIR_BOOK2),
        )
    for mapping, directory in search:
        filename = mapping.get(canonical_name)
        if filename:
            path = directory / filename
            if path.exists():
                return path
    return None


def character_asset_note(character_name, book=None) -> str:
    """Human-readable note for prompts and saved metadata."""
    canonical_name = canonical_character_name(character_name)
    if not canonical_name:
        return "No character portrait selected."
    portrait = resolve_character_portrait_asset(canonical_name, book=book)
    name_mark = resolve_character_name_asset(canonical_name)
    parts = [canonical_name]
    if portrait:
        parts.append(f"portrait: {portrait}")
    if name_mark:
        parts.append(f"name mark: {name_mark}")
    return " | ".join(parts)
