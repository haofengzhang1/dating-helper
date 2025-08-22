AI Dating Helper
================

AI Dating Helper is a lightweight tool built with FastAPI (backend) and vanilla HTML/JS (frontend), designed around three core features: chatting assistant (generate smooth replies and conversation ideas), EQ test (score emotional intelligence based on answers and provide explanations), and restaurant recommendations (suggest date spots based on location and budget). The backend provides three simple API groups: /api/chat for dialogue support (powered by Groq’s Llama 3.1 by default), /api/eq/score for EQ test scoring, and /api/restaurants for retrieving restaurant lists (via Yelp or Google Places). The frontend renders these results directly into suggested messages, EQ reports, and clickable restaurant cards.

Use
--------
Run the backend with uvicorn main:app --reload --port 8000. For the frontend, simply serve the static files locally (e.g., python -m http.server 5500) and open http://127.0.0.1:5500 in the browser. API usage is simple: POST /api/chat (params: message, optional context) returns conversation suggestions, POST /api/eq/score (params: answers array) returns EQ score and interpretation, and GET /api/restaurants (params like query, lat, lng, budget, open_now) returns restaurants with name, rating, price, address, and link.
