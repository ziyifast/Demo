ROLE_RANK = {"agent": 1, "supervisor": 2, "finance": 3}


def required_level(amount: float) -> str:
    if amount <= 100:
        return "agent"
    if amount <= 1000:
        return "supervisor"
    return "finance"


def can_approve(role: str, amount: float) -> bool:
    return ROLE_RANK.get(role, 0) >= ROLE_RANK[required_level(amount)]