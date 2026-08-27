import re

# Category keyword and regex pattern rules
# Target taxonomy: Food, Transport, Entertainment, Utilities, Shopping, Other

RULES = {
    'Food': [
        r'\bswiggy\b', r'\bzomato\b', r'\bdominos\b', r'\bmcdonalds?\b', r'\bstarbucks\b',
        r'\bcafe\b', r'\bfood\b', r'\brestaurant\b', r'\bpizza\b', r'\bburger\b',
        r'\bdining\b', r'\bgroceries\b', r'\bgrocery\b', r'\beat\b', r'\bdrink\b',
        r'\bbakery\b', r'\blunch\b', r'\bdinner\b', r'\bbreakfast\b', r'\bkitchen\b',
        r'\bwater\b', r'\btea\b', r'\bcoffee\b', r'\bjuice\b', r'\bsnacks?\b'
    ],
    'Transport': [
        r'\buber\b', r'\bola\b', r'\bpetrol\b', r'\bdiesel\b', r'\bfuel\b',
        r'\bmetro\b', r'\bcab\b', r'\btaxi\b', r'\bflight\b', r'\bairline\b',
        r'\btrain\b', r'\birctc\b', r'\bparking\b', r'\bbus\b', r'\btravel\b',
        r'\btrip\b', r'\btransport\b', r'\btoll\b', r'\bvehicle\b', r'\bauto\b',
        r'\bcommute\b', r'\brace\b', r'\bdrive\b', r'\bcar\b', r'\bbike\b'
    ],
    'Entertainment': [
        r'\bnetflix\b', r'\bspotify\b', r'\bcinema\b', r'\bmovie\b', r'\btheater\b',
        r'\bprime\b', r'\bhotstar\b', r'\bgame\b', r'\bgaming\b', r'\bconcert\b',
        r'\bshow\b', r'\bevent\b', r'\bticket\b', r'\bplay\b', r'\bfun\b',
        r'\bclub\b', r'\bparty\b', r'\bmusic\b', r'\bfestival\b'
    ],
    'Utilities': [
        r'\belectricity\b', r'\bwater bill\b', r'\bgas\b', r'\bbill\b', r'\brent\b',
        r'\brecharge\b', r'\bwifi\b', r'\bbroadband\b', r'\bphone\b', r'\bpower\b',
        r'\butility\b', r'\butilities\b', r'\btenant\b', r'\blease\b', r'\bmaintenance\b',
        r'\bhouse\b', r'\binternet\b', r'\bdues\b', r'\bfee\b', r'\brent\b'
    ],
    'Shopping': [
        r'\bamazon\b', r'\bflipkart\b', r'\bmyntra\b', r'\bclothes\b', r'\bfashion\b',
        r'\bshoes\b', r'\bstore\b', r'\bmall\b', r'\bsupermarket\b', r'\bretail\b',
        r'\bpurchase\b', r'\bbuy\b', r'\belectronics\b', r'\bshop\b', r'\bapparel\b',
        r'\border\b', r'\bmarket\b', r'\boutfit\b', r'\bgadget\b'
    ],
    'Other': [
        r'\bhealth\b', r'\bfitness\b', r'\bdoctor\b', r'\bhospital\b', r'\bmedicine\b',
        r'\bpharmacy\b', r'\bsalary\b', r'\binvestment\b', r'\btax\b', r'\binsurance\b',
        r'\bfee\b', r'\bdonation\b', r'\bgift\b', r'\bother\b'
    ]
}

# Compile regex patterns once for optimal performance
COMPILED_RULES = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in RULES.items()
}

def predict_category(text, amount=None):
    """
    Predicts transaction category using baseline regex and keyword matching.
    
    Args:
        text (str): Transaction description or title.
        amount (float, optional): Transaction amount (for future rule heuristics).
        
    Returns:
        tuple: (predicted_category, confidence_score)
    """
    if not text or not isinstance(text, str):
        return ("Other", 0.1)

    normalized_text = text.strip()

    # Iterate through compiled rule sets
    for category, compiled_patterns in COMPILED_RULES.items():
        for pattern in compiled_patterns:
            if pattern.search(normalized_text):
                # Return match with high confidence (0.95 for exact rule match)
                return (category, 0.95)

    # Fallback default category
    return ("Other", 0.10)

if __name__ == '__main__':
    # Simple test driver
    test_samples = [
        "Swiggy order lunch",
        "Uber ride to office",
        "Netflix monthly subscription",
        "Electricity bill payment",
        "Amazon purchase new shoes",
        "Random transaction description"
    ]
    print("--- Categorizer Rules Baseline Test ---")
    for sample in test_samples:
        cat, conf = predict_category(sample)
        print(f"Text: '{sample}' -> Category: {cat} (Confidence: {conf})")
