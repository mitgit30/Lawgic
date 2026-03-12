import re

INDIAN_STATES_AND_UTS = [
    "andhra pradesh",
    "arunachal pradesh",
    "assam",
    "bihar",
    "chhattisgarh",
    "goa",
    "gujarat",
    "haryana",
    "himachal pradesh",
    "jharkhand",
    "karnataka",
    "kerala",
    "madhya pradesh",
    "maharashtra",
    "manipur",
    "meghalaya",
    "mizoram",
    "nagaland",
    "odisha",
    "punjab",
    "rajasthan",
    "sikkim",
    "tamil nadu",
    "telangana",
    "tripura",
    "uttar pradesh",
    "uttarakhand",
    "west bengal",
    "andaman and nicobar islands",
    "chandigarh",
    "dadra and nagar haveli and daman and diu",
    "delhi",
    "jammu and kashmir",
    "ladakh",
    "lakshadweep",
    "puducherry",
]


def infer_state(text: str | None) -> str | None:
    if not text:
        return None
    normalized = text.lower()
    for state in INDIAN_STATES_AND_UTS:
        if re.search(rf"\b{re.escape(state)}\b", normalized):
            return state
    return None


def detect_chunk_state(metadata: dict, text: str) -> str | None:
    for key in ("state", "jurisdiction_state", "state_name"):
        value = metadata.get(key)
        if isinstance(value, str):
            state = infer_state(value)
            if state:
                return state
    return infer_state(text)


def is_central_reference(metadata: dict, text: str) -> bool:
    for key in ("authority_level", "jurisdiction_level", "reference_scope"):
        value = str(metadata.get(key, "")).lower()
        if "central" in value or "union" in value:
            return True
    normalized = text.lower()
    return "parliament" in normalized or "constitution of india" in normalized
