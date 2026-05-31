"""Hand-crafted Chain-of-Thought exemplars for table question answering.

Each exemplar is a dict with keys:
    question : str
    table    : {"header": [str, ...], "rows": [[str, ...], ...]}
    reasoning: str
    answer   : str

Two parallel lists are provided:
    EXEMPLARS_PLAIN      -- natural-language paragraph reasoning
    EXEMPLARS_STRUCTURED -- numbered step-by-step reasoning

Every answer has been independently verified against its table.
"""

from __future__ import annotations

from .data import serialize_table

# ---------------------------------------------------------------------------
# EXEMPLARS_PLAIN
# ---------------------------------------------------------------------------
EXEMPLARS_PLAIN: list[dict] = [
    # 1. Simple cell lookup
    {
        "question": "What is Alice's score?",
        "table": {
            "header": ["Name", "Score"],
            "rows": [
                ["Alice", "92"],
                ["Bob", "78"],
                ["Carol", "85"],
            ],
        },
        "reasoning": (
            "I need to find the row where the Name column equals 'Alice'. "
            "Looking through the rows, the first row has Name = 'Alice' and Score = '92'. "
            "Therefore Alice's score is 92."
        ),
        "answer": "92",
    },
    # 2. Row filtering by condition
    {
        "question": "Which city has a population greater than 2 million?",
        "table": {
            "header": ["City", "Population (millions)"],
            "rows": [
                ["Springfield", "0.5"],
                ["Shelbyville", "2.3"],
                ["Capital City", "1.1"],
            ],
        },
        "reasoning": (
            "I need to filter rows where the population is greater than 2 million. "
            "Springfield has 0.5 million, which is not greater than 2. "
            "Shelbyville has 2.3 million, which is greater than 2, so it qualifies. "
            "Capital City has 1.1 million, which is not greater than 2. "
            "Only Shelbyville satisfies the condition."
        ),
        "answer": "Shelbyville",
    },
    # 3. Counting
    {
        "question": "How many products cost more than $20?",
        "table": {
            "header": ["Product", "Price"],
            "rows": [
                ["Widget", "15"],
                ["Gadget", "25"],
                ["Doohickey", "30"],
                ["Thingamajig", "10"],
            ],
        },
        "reasoning": (
            "I need to count how many rows have Price greater than 20. "
            "Widget costs 15, which is not more than 20. "
            "Gadget costs 25, which is more than 20. "
            "Doohickey costs 30, which is more than 20. "
            "Thingamajig costs 10, which is not more than 20. "
            "Two products (Gadget and Doohickey) cost more than $20."
        ),
        "answer": "2",
    },
    # 4. Aggregation: sum
    {
        "question": "What is the total revenue across all months?",
        "table": {
            "header": ["Month", "Revenue"],
            "rows": [
                ["January", "1200"],
                ["February", "1500"],
                ["March", "1100"],
            ],
        },
        "reasoning": (
            "I need to sum the Revenue column for all rows. "
            "January contributes 1200, February contributes 1500, and March contributes 1100. "
            "Adding these together: 1200 + 1500 = 2700, then 2700 + 1100 = 3800. "
            "The total revenue is 3800."
        ),
        "answer": "3800",
    },
    # 5. Aggregation: max
    {
        "question": "What is the highest temperature recorded?",
        "table": {
            "header": ["City", "High Temp (F)"],
            "rows": [
                ["Phoenix", "118"],
                ["Las Vegas", "115"],
                ["Death Valley", "130"],
                ["Tucson", "112"],
            ],
        },
        "reasoning": (
            "I need to find the maximum value in the High Temp (F) column. "
            "The temperatures are 118 for Phoenix, 115 for Las Vegas, 130 for Death Valley, "
            "and 112 for Tucson. "
            "The highest among these is 130, recorded in Death Valley."
        ),
        "answer": "130",
    },
    # 6. Comparison
    {
        "question": "Does team Lions have more wins than team Tigers?",
        "table": {
            "header": ["Team", "Wins", "Losses"],
            "rows": [
                ["Lions", "10", "4"],
                ["Tigers", "7", "7"],
                ["Bears", "10", "4"],
            ],
        },
        "reasoning": (
            "I need to compare the Wins values for Lions and Tigers. "
            "Looking up Lions, they have 10 wins. "
            "Looking up Tigers, they have 7 wins. "
            "Since 10 is greater than 7, Lions have more wins than Tigers."
        ),
        "answer": "yes",
    },
    # 7. Sorting / superlative: min
    {
        "question": "Which country has the smallest area?",
        "table": {
            "header": ["Country", "Area (sq km)"],
            "rows": [
                ["Brazil", "8515767"],
                ["Canada", "9984670"],
                ["Australia", "7692024"],
                ["India", "3287263"],
            ],
        },
        "reasoning": (
            "I need to find the country with the minimum value in the Area (sq km) column. "
            "The areas are: Brazil 8,515,767; Canada 9,984,670; Australia 7,692,024; India 3,287,263. "
            "Comparing all four, India's area of 3,287,263 is the smallest. "
            "Therefore India has the smallest area."
        ),
        "answer": "India",
    },
    # 8. Multi-hop: filter to cheapest, then look up its rating
    {
        "question": "What is the rating of the cheapest product?",
        "table": {
            "header": ["Product", "Price", "Rating"],
            "rows": [
                ["Alpha", "50", "4.2"],
                ["Beta", "30", "3.8"],
                ["Gamma", "45", "4.5"],
            ],
        },
        "reasoning": (
            "This is a two-step question. "
            "First, I find the cheapest product by looking at the Price column: "
            "Alpha costs 50, Beta costs 30, and Gamma costs 45. "
            "The minimum price is 30, so the cheapest product is Beta. "
            "Second, I look up Beta's rating, which is 3.8. "
            "Therefore the rating of the cheapest product is 3.8."
        ),
        "answer": "3.8",
    },
]

# ---------------------------------------------------------------------------
# EXEMPLARS_STRUCTURED
# ---------------------------------------------------------------------------
EXEMPLARS_STRUCTURED: list[dict] = [
    # 1. Simple cell lookup
    {
        "question": "What is Alice's score?",
        "table": {
            "header": ["Name", "Score"],
            "rows": [
                ["Alice", "92"],
                ["Bob", "78"],
                ["Carol", "85"],
            ],
        },
        "reasoning": (
            "Step 1: Identify the target column. The question asks for a Score value.\n"
            "Step 2: Filter rows by Name = 'Alice'. The matching row is ['Alice', '92'].\n"
            "Step 3: Read the Score value from that row: 92.\n"
            "Conclusion: Alice's score is 92."
        ),
        "answer": "92",
    },
    # 2. Row filtering by condition
    {
        "question": "Which city has a population greater than 2 million?",
        "table": {
            "header": ["City", "Population (millions)"],
            "rows": [
                ["Springfield", "0.5"],
                ["Shelbyville", "2.3"],
                ["Capital City", "1.1"],
            ],
        },
        "reasoning": (
            "Step 1: Identify the filtering condition: Population (millions) > 2.\n"
            "Step 2: Check each row.\n"
            "  - Springfield: 0.5 > 2? No.\n"
            "  - Shelbyville: 2.3 > 2? Yes.\n"
            "  - Capital City: 1.1 > 2? No.\n"
            "Step 3: Collect cities that pass: Shelbyville.\n"
            "Conclusion: Shelbyville is the only city with a population greater than 2 million."
        ),
        "answer": "Shelbyville",
    },
    # 3. Counting
    {
        "question": "How many products cost more than $20?",
        "table": {
            "header": ["Product", "Price"],
            "rows": [
                ["Widget", "15"],
                ["Gadget", "25"],
                ["Doohickey", "30"],
                ["Thingamajig", "10"],
            ],
        },
        "reasoning": (
            "Step 1: Identify the filtering condition: Price > 20.\n"
            "Step 2: Check each row.\n"
            "  - Widget: 15 > 20? No.\n"
            "  - Gadget: 25 > 20? Yes. (count = 1)\n"
            "  - Doohickey: 30 > 20? Yes. (count = 2)\n"
            "  - Thingamajig: 10 > 20? No.\n"
            "Step 3: Total count of qualifying rows = 2.\n"
            "Conclusion: 2 products cost more than $20."
        ),
        "answer": "2",
    },
    # 4. Aggregation: sum
    {
        "question": "What is the total revenue across all months?",
        "table": {
            "header": ["Month", "Revenue"],
            "rows": [
                ["January", "1200"],
                ["February", "1500"],
                ["March", "1100"],
            ],
        },
        "reasoning": (
            "Step 1: Identify the column to sum: Revenue.\n"
            "Step 2: Collect all Revenue values: 1200, 1500, 1100.\n"
            "Step 3: Add them: 1200 + 1500 = 2700; 2700 + 1100 = 3800.\n"
            "Conclusion: The total revenue is 3800."
        ),
        "answer": "3800",
    },
    # 5. Aggregation: max
    {
        "question": "What is the highest temperature recorded?",
        "table": {
            "header": ["City", "High Temp (F)"],
            "rows": [
                ["Phoenix", "118"],
                ["Las Vegas", "115"],
                ["Death Valley", "130"],
                ["Tucson", "112"],
            ],
        },
        "reasoning": (
            "Step 1: Identify the column to maximize: High Temp (F).\n"
            "Step 2: Collect all temperature values: 118, 115, 130, 112.\n"
            "Step 3: Find the maximum: 130 (Death Valley).\n"
            "Conclusion: The highest temperature recorded is 130."
        ),
        "answer": "130",
    },
    # 6. Comparison
    {
        "question": "Does team Lions have more wins than team Tigers?",
        "table": {
            "header": ["Team", "Wins", "Losses"],
            "rows": [
                ["Lions", "10", "4"],
                ["Tigers", "7", "7"],
                ["Bears", "10", "4"],
            ],
        },
        "reasoning": (
            "Step 1: Look up Wins for Lions: 10.\n"
            "Step 2: Look up Wins for Tigers: 7.\n"
            "Step 3: Compare: 10 > 7, so Lions have more wins.\n"
            "Conclusion: Yes, Lions have more wins than Tigers."
        ),
        "answer": "yes",
    },
    # 7. Superlative: min
    {
        "question": "Which country has the smallest area?",
        "table": {
            "header": ["Country", "Area (sq km)"],
            "rows": [
                ["Brazil", "8515767"],
                ["Canada", "9984670"],
                ["Australia", "7692024"],
                ["India", "3287263"],
            ],
        },
        "reasoning": (
            "Step 1: Identify the column to minimize: Area (sq km).\n"
            "Step 2: Collect all area values: 8515767, 9984670, 7692024, 3287263.\n"
            "Step 3: Find the minimum: 3287263 (India).\n"
            "Conclusion: India has the smallest area."
        ),
        "answer": "India",
    },
    # 8. Multi-hop: cheapest product's rating
    {
        "question": "What is the rating of the cheapest product?",
        "table": {
            "header": ["Product", "Price", "Rating"],
            "rows": [
                ["Alpha", "50", "4.2"],
                ["Beta", "30", "3.8"],
                ["Gamma", "45", "4.5"],
            ],
        },
        "reasoning": (
            "Step 1: Find the cheapest product by minimizing the Price column.\n"
            "  - Alpha: 50, Beta: 30, Gamma: 45. Minimum price = 30 (Beta).\n"
            "Step 2: Look up the Rating for Beta: 3.8.\n"
            "Conclusion: The rating of the cheapest product (Beta) is 3.8."
        ),
        "answer": "3.8",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_exemplars(style: str) -> list[dict]:
    """Return the exemplar list for the given style ("plain" or "structured")."""
    if style == "plain":
        return EXEMPLARS_PLAIN
    if style == "structured":
        return EXEMPLARS_STRUCTURED
    raise ValueError(f"Unknown style {style!r}. Choose 'plain' or 'structured'.")


def format_exemplar(ex: dict, style: str) -> str:  # noqa: ARG001  (style is informational)
    """Format one exemplar as a demonstration string ending in "Answer: <answer>".

    style is informational only; each exemplar already carries the reasoning
    text for its list, so it is not used to render.
    """
    serialized = serialize_table(ex["table"])
    lines = [
        f"Question: {ex['question']}",
        "Table:",
        serialized,
        f"Reasoning: {ex['reasoning']}",
        f"Answer: {ex['answer']}",
    ]
    return "\n".join(lines)
