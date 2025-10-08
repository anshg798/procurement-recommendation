import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from serpapi import GoogleSearch
from groq import Groq

# Load environment variables
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize clients
client = Groq(api_key=GROQ_API_KEY)

# FastAPI app
app = FastAPI(
    title="Procurement Recommendation API",
    description="AI-powered procurement recommendation system using real-time data from SERP and Llama-3.1-8b-instant model.",
    version="1.0"
)


# Input schema
class ProcurementRequest(BaseModel):
    material_name: str
    quantity: int
    location: str
    budget: float


# Utility: Fetch real-time supplier data using SERP API
def get_supplier_data(material: str, location: str):
    try:
        query = f"{material} suppliers in {location}"
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 5
        })
        results = search.get_dict()
        organic = results.get("organic_results", [])
        suppliers = []

        for res in organic[:5]:
            suppliers.append({
                "title": res.get("title"),
                "link": res.get("link"),
                "snippet": res.get("snippet")
            })

        return suppliers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SERPAPI error: {str(e)}")


# Utility: Generate procurement recommendation using Llama model
def generate_recommendation(material, quantity, location, budget, suppliers):
    supplier_summary = "\n".join(
        [f"- {s['title']}: {s['link']} ({s['snippet']})" for s in suppliers]
    )

    prompt = f"""
    You are an AI procurement strategist for POWERGRID.
    Analyze the following inputs and recommend a procurement plan.

    Material: {material}
    Required Quantity: {quantity}
    Project Location: {location}
    Budget: ₹{budget}

    Available supplier data:
    {supplier_summary}

    Provide:
    1. Top 3 suppliers ranked by relevance.
    2. Recommended order quantity split (if applicable).
    3. Estimated total cost and delivery timeframe.
    4. Any risk factors or negotiation suggestions.
    """

    chat_completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert AI procurement assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=800,
    )

    return chat_completion.choices[0].message.content


# API endpoint
@app.post("/recommend-procurement")
def recommend_procurement(request: ProcurementRequest):
    suppliers = get_supplier_data(request.material_name, request.location)
    if not suppliers:
        raise HTTPException(status_code=404, detail="No suppliers found.")

    recommendation = generate_recommendation(
        request.material_name,
        request.quantity,
        request.location,
        request.budget,
        suppliers
    )

    return {
        "material": request.material_name,
        "location": request.location,
        "budget": request.budget,
        "recommendation": recommendation,
        "top_suppliers": suppliers
    }
