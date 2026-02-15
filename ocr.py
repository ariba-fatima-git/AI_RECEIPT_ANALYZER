import base64
import re
import os
import cv2
from groq import Groq
from groq import Client  
from preprocess import preprocess_image 

from dotenv import load_dotenv

# Load API key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_text_from_image(image_path):
    image_base64 = encode_image(image_path)

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this receipt."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    image_path = "receipt.jpg"  # make sure this file exists
    processed_image = preprocess_image(image_path)
   # text = extract_text_from_image(processed_image)


   # print("\nExtracted Text:\n")
   # print(text)

def preprocess_image(input_path, output_path="processed.jpg"):
    # Read image
    img = cv2.imread(input_path)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Noise reduction (Gaussian blur)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Contrast enhancement using adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Save processed image
    cv2.imwrite(output_path, thresh)

    return output_path

#step2
def parse_receipt_text(ocr_text):
    """
    Converts OCR text into structured data:
    [{"item": "...", "price": ...}, ...] and extracts total separately
    """
    items = []
    total = None

    # Split text into lines
    lines = ocr_text.split("\n")

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Detect TOTAL line and extract total value
        total_match = re.search(r"TOTAL\s*\$?([\d]+\.?[\d]*)", line, re.IGNORECASE)
        if total_match:
            total = float(total_match.group(1))
            continue  # skip adding TOTAL as an item

        # Match normal items with price
        match = re.search(r"([A-Za-z\s]+)\s+\$?([\d]+\.?[\d]*)", line)
        if match:
            item_name = match.group(1).strip()
            price = float(match.group(2))
            items.append({"item": item_name, "price": price})

    # Add total as separate entry
    if total is not None:
        items.append({"total": total})

    return items


if __name__ == "__main__":
    # Example Groq OCR text
    sample_text = """
Orange Juice $2.15
Apples $3.50
Tomato $2.40
Fish $6.99
Beef $10.00
Onion $1.25
Cheese $3.40
TOTAL $29.69
"""
    
#step3
# categorizer.py

def categorize_items(items):
    """
    Assign categories to each item and calculate totals per category.
    Input: items = [{"item": "Orange Juice", "price": 2.15}, ...]
    Output: {
        "categorized_items": [{"item": ..., "price": ..., "category": ...}, ...],
        "category_totals": {"fruit": 5.90, "meat": 16.99, ...},
        "overall_total": 29.69
    }
    """

    # Define simple category keywords
    category_keywords = {
        "fruit": ["apple", "orange", "banana", "tomato", "onion"],
        "meat": ["beef", "chicken", "fish", "pork"],
        "dairy": ["cheese", "milk", "yogurt", "butter"],
        "beverage": ["juice", "cola", "water", "coffee", "tea"],
        "bakery": ["bread", "cake", "cookie", "bun"],
        "other": []  # fallback category
    }

    categorized_items = []
    category_totals = {cat: 0.0 for cat in category_keywords.keys()}
    overall_total = 0.0

    for item in items:
        # Skip the total entry
        if "total" in item:
            overall_total = item["total"]
            continue

        item_name = item["item"].lower()
        price = item["price"]

        # Default category
        item_category = "other"

        # Find category based on keywords
        for category, keywords in category_keywords.items():
            for kw in keywords:
                if kw in item_name:
                    item_category = category
                    break
            if item_category != "other":
                break

        # Add category to item
        categorized_items.append({
            "item": item["item"],
            "price": price,
            "category": item_category
        })

        # Update category total
        category_totals[item_category] += price
        overall_total = sum(category_totals.values())

    return {
        "categorized_items": categorized_items,
        "category_totals": category_totals,
        "overall_total": overall_total
    }


if __name__ == "__main__":
    # Example structured data from parser
    items = [
        {'item': 'Orange Juice', 'price': 2.15},
        {'item': 'Apples', 'price': 3.5},
        {'item': 'Tomato', 'price': 2.4},
        {'item': 'Fish', 'price': 6.99},
        {'item': 'Beef', 'price': 10.0},
        {'item': 'Onion', 'price': 1.25},
        {'item': 'Cheese', 'price': 3.4},
        {'total': 29.69}
    ]

    

# Step 5: Spending Analysis
# ===========================
def analyze_spending(category_data, overspend_threshold=0.3):
    """
    Calculate percentage of total spending per category and flag overspending.
    overspend_threshold: fraction of total to flag (default 30%)
    """
    overall = category_data["overall_total"]
    analysis = {}
    for category, total in category_data["category_totals"].items():
        if overall == 0:
            percent = 0
        else:
            percent = (total / overall) * 100
        overspend = percent > (overspend_threshold * 100)
        analysis[category] = {
            "total": total,
            "percent": round(percent, 2),
            "overspend": overspend
        }
    return analysis

# ===========================
# Step 6: LLM Insights
# ===========================

def generate_budget_advice(categorized_data):
    """
    Feed categorized spending data to LLM and get personalized advice.
    """
    # Prepare a structured prompt for the LLM
    prompt = "You are a personal finance assistant.\n"
    prompt += "Here is a user's monthly shopping receipt breakdown:\n"

    for item in categorized_data["categorized_items"]:
        prompt += f"- {item['item']}: ${item['price']} ({item['category']})\n"

    prompt += "\nOverall spending by category:\n"
    for cat, total in categorized_data["category_totals"].items():
        prompt += f"- {cat}: ${total}\n"

    prompt += f"\nTotal: ${categorized_data['overall_total']}\n"
    prompt += "\nProvide actionable advice to save money, avoid overspending, and budget effectively."

    # Create Groq client
    client = Client(api_key=os.getenv("GROQ_API_KEY")) # replace with your key

    # Call the LLM
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": "You are a financial advisor."},
            {"role": "user", "content": prompt}
        ]
    )

    advice_text = completion.choices[0].message.content
    return advice_text


# ===========================
# Main Execution
# ===========================
if __name__ == "__main__":
    # Step 1: Preprocess the receipt
    processed_image = preprocess_image("receipt.jpg")

    # Step 2: Extract text via Groq OCR
    ocr_text = extract_text_from_image(processed_image)
    print("OCR Text:\n", ocr_text)

    # Step 3: Parse OCR text
    structured_items = parse_receipt_text(ocr_text)
    print("\nStructured Items:\n", structured_items)

    # Step 4: Categorize items
    categorized_data = categorize_items(structured_items)

    # Fix overall total
    categorized_data["overall_total"] = sum(categorized_data["category_totals"].values())

    spending_analysis = analyze_spending(categorized_data)

    
    print("\nCategorized Data:\n", categorized_data)

    # Step 5: Spending Analysis
    spending_analysis = analyze_spending(categorized_data)
    print("\nSpending Analysis:\n")
    for cat, data in spending_analysis.items():
        flag = "⚠️ Overspending!" if data["overspend"] else ""
        print(f"{cat.capitalize():<10}: ${data['total']:.2f} ({data['percent']}%) {flag}")
        
    # Step 6: LLM Insights
    budget_advice = generate_budget_advice(categorized_data)
    print("\n💡 Personalized Budget Advice:\n")
    print(budget_advice)
    
        
        