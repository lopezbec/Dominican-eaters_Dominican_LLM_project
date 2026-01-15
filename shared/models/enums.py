from enum import Enum


class ContentAvailability(str, Enum):
    NOT_FOUND = "NO ENCONTRADO"
    FOUND = "ENCONTRADO"
    PARTIAL = "PARCIAL"
