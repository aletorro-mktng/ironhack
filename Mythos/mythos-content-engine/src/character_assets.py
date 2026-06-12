"""Shared Mortal Vengeance character asset registry."""

from __future__ import annotations

import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHARACTER_NAME_DIR = PROJECT_ROOT / "assets" / "character_names"
CHARACTER_PORTRAIT_DIR = PROJECT_ROOT / "assets" / "character_portraits" / "book1"


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
    if text in CHARACTER_NAME_ASSETS or text in BOOK1_CHARACTER_PORTRAITS:
        return text
    return _ASSET_ALIASES.get(normalize_asset_key(text), text)


def resolve_character_name_asset(character_name) -> Path | None:
    """Return the uploaded character-name mark for a selected character."""
    canonical_name = canonical_character_name(character_name)
    filename = CHARACTER_NAME_ASSETS.get(canonical_name)
    if not filename:
        return None
    path = CHARACTER_NAME_DIR / filename
    return path if path.exists() else None


def resolve_character_portrait_asset(character_name) -> Path | None:
    """Return the Book 1 portrait for a selected character, when available."""
    canonical_name = canonical_character_name(character_name)
    filename = BOOK1_CHARACTER_PORTRAITS.get(canonical_name)
    if not filename:
        return None
    path = CHARACTER_PORTRAIT_DIR / filename
    return path if path.exists() else None


def character_asset_note(character_name) -> str:
    """Human-readable note for prompts and saved metadata."""
    canonical_name = canonical_character_name(character_name)
    if not canonical_name:
        return "No character portrait selected."
    portrait = resolve_character_portrait_asset(canonical_name)
    name_mark = resolve_character_name_asset(canonical_name)
    parts = [canonical_name]
    if portrait:
        parts.append(f"portrait: {portrait}")
    if name_mark:
        parts.append(f"name mark: {name_mark}")
    return " | ".join(parts)
