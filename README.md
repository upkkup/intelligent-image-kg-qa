# Intelligent Image Understanding & Knowledge Graph QA

React/Vite frontend with a FastAPI backend, YOLO object detection, Wikidata entity search/SPARQL retrieval, and optional OpenAI-compatible LLM synthesis.

## Local run
- API: `cd api`, install `requirements.txt`, then run `uvicorn app.main:app --reload --port 8000`.
- UI: `cd web`, run `npm install` and `npm run dev`.
- Docker: `docker compose up --build`, then open http://localhost:8080.

## Deploy
In Render, choose **New → Blueprint**, connect this repository, and select branch `deploy/online-deployment`. Render uses `render.yaml`. Test `/api/health` on the generated service URL before considering deployment successful.

Set `LLM_BASE_URL`, `LLM_API_KEY` (secret), and `LLM_MODEL` in service environment variables for LLM synthesis. Never commit API keys.

## Research limitations
Spatial relations are heuristics, not a trained visual relationship model. The current live knowledge backend is Wikidata; Freebase/Wikipedia connectors and rigorous benchmark evaluation remain future work. Add authentication, rate limiting, monitoring, and suitable compute capacity before production use.
