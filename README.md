# Intelligent Image Understanding & Knowledge Graph QA

A research prototype that combines image understanding, structured knowledge retrieval, SPARQL, and optional LLM reasoning.

## Implemented components
- **Frontend:** React + TypeScript + Vite.
- **Object detection:** YOLO through Ultralytics; returns class, confidence, bounding box, dimensions, and overlays boxes on the uploaded image.
- **Scene relations:** transparent spatial heuristics. These are not predictions from a trained visual-relationship detector.
- **Knowledge retrieval:** Wikidata entity search and SPARQL.
- **Question templates:** common forms such as capital, population, official language, currency, who-founded, and when-founded are mapped to Wikidata properties when detected. Other questions use general entity/fact retrieval.
- **LLM synthesis:** optional OpenAI-compatible endpoint. Evidence is passed separately from the question, and the API reports whether an LLM response was actually used.
- **API:** FastAPI endpoints: /api/health, /api/analyze-image, and /api/qa.
- **CI:** unit tests mock remote network and model dependencies; frontend TypeScript/Vite build is checked in GitHub Actions.

## Local run
Requirements: Python 3.11+, Node.js 20+, and adequate RAM for CPU inference.

API terminal:
~~~bash
cd api
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
~~~

Frontend terminal:
~~~bash
cd web
npm install
npm run dev
~~~

The Vite development server proxies /api to http://localhost:8000.

## Docker
Run docker compose up --build and open http://localhost:8080. The container installs CPU-only PyTorch. The first object-detection request may download YOLO model weights.

## Environment configuration
Copy .env.example to .env for local Docker use, then set:
- LLM_BASE_URL: OpenAI-compatible API base URL.
- LLM_API_KEY: API key; do not commit real secrets.
- LLM_MODEL: model ID required for LLM synthesis.
- YOLO_MODEL: defaults to yolov8n.pt.
- WIKIDATA_USER_AGENT: descriptive user agent for Wikidata requests.

If LLM credentials are not configured, the service returns an explicit evidence/retrieval fallback rather than claiming to have used an LLM.

## Deployment (Render)
1. Review the deployment branch and CI results.
2. In Render, create a Blueprint and connect this repository using branch deploy/online-deployment.
3. Configure LLM environment variables in the service's secret/environment settings.
4. Deploy, then verify /api/health, image upload, and question answering.

The Render Blueprint specifies a paid Starter plan; confirm the cost before deploying. No live deployment is claimed until a real service URL has been tested.

## Current limitations / next milestones
- Wikidata is the only live knowledge source; Freebase, Wikipedia ingestion, and local graph/corpus connectors are not implemented yet.
- The current template parser is rule-based, not a general semantic parser. Embedding retrieval and learned question classifiers remain future work.
- Spatial relations are geometric heuristics; a trained visual relationship detector is still needed.
- The first entity returned by Wikidata search is used; entity disambiguation and retrieval evaluation need a labeled benchmark.
- Production hardening should include rate limiting, authentication where appropriate, request/latency logging, caching, and resource limits.
- Evaluate accuracy, entity-linking quality, SPARQL execution success, answer faithfulness, latency, and cost before research claims.
