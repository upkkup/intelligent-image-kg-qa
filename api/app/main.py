from __future__ import annotations
import io, os, re
from pathlib import Path
from typing import Any
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

app = FastAPI(title="Intelligent Image & Knowledge Graph QA", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","), allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
MAX_IMAGE_BYTES = 10 * 1024 * 1024
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = os.getenv("WIKIDATA_USER_AGENT", "IntelligentImageKGQA/0.1 (research prototype)")
STATIC_DIR = Path(__file__).parent / "static"
_yolo_model = None

class QARequest(BaseModel):
    question: str = Field(min_length=2, max_length=1500)
    image_context: dict[str, Any] | None = None
    model: str | None = None

def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(os.getenv("YOLO_MODEL", "yolov8n.pt"))
    return _yolo_model

def _scene_relations(detections):
    # Transparent spatial heuristic, not a trained visual relationship model.
    relations = []
    for i, a in enumerate(detections):
        ax1, ay1, ax2, ay2 = a["bbox"]
        acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        for b in detections[i + 1:]:
            bx1, by1, bx2, by2 = b["bbox"]
            bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if ax1 <= bcx <= ax2 and ay1 <= bcy <= ay2: relation = "contains/overlaps"
            elif abs(acy - bcy) < abs(acx - bcx): relation = "left of" if acx < bcx else "right of"
            else: relation = "above" if acy < bcy else "below"
            relations.append({"subject": a["label"], "relation": relation, "object": b["label"]})
    return relations[:100]

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "intelligent-image-kg-qa", "llm_configured": bool(os.getenv("LLM_API_KEY") and os.getenv("LLM_BASE_URL"))}

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(415, "Upload a JPEG, PNG, WEBP, or BMP image.")
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image must be non-empty and at most 10 MB.")
    try: image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc: raise HTTPException(400, "The uploaded file is not a valid image.") from exc
    try:
        result = _load_yolo().predict(image, verbose=False, conf=0.25)[0]
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = [round(float(x), 2) for x in box.xyxy[0].tolist()]
            detections.append({"label": result.names.get(cls_id, str(cls_id)), "confidence": round(float(box.conf[0].item()), 4), "bbox": [x1,y1,x2,y2], "width": round(x2-x1,2), "height": round(y2-y1,2)})
        labels = ", ".join(d["label"] for d in detections) or "no objects detected"
        return {"filename": file.filename, "image_size": {"width": image.width, "height": image.height}, "detections": detections, "relations": _scene_relations(detections), "description": f"The image contains {labels}. Spatial relations are heuristic estimates."}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(503, f"Image analysis failed; check model installation/resources ({type(exc).__name__}).") from exc

def wikidata_search(term: str):
    r = requests.get(WIKIDATA_API, params={"action":"wbsearchentities","search":term,"language":"en","format":"json","limit":5}, headers={"User-Agent":USER_AGENT}, timeout=12)
    r.raise_for_status()
    return [{"id":x["id"],"label":x.get("label",""),"description":x.get("description","")} for x in r.json().get("search",[])]

def wikidata_facts(entity_id: str):
    if not re.fullmatch(r"Q[1-9][0-9]*", entity_id): raise ValueError("Invalid Wikidata entity ID")
    query = f'''
SELECT ?propertyLabel ?valueLabel WHERE {{
  ?property wikibase:directClaim ?p .
  wd:{entity_id} ?p ?value .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
  FILTER(isIRI(?value))
}} LIMIT 15
'''
    r = requests.get(SPARQL_ENDPOINT, params={"query":query,"format":"json"}, headers={"User-Agent":USER_AGENT,"Accept":"application/sparql-results+json"}, timeout=18)
    r.raise_for_status()
    rows = r.json().get("results",{}).get("bindings",[])
    facts = [{"property":x.get("propertyLabel",{}).get("value",""),"value":x.get("valueLabel",{}).get("value","")} for x in rows]
    return query.strip(), facts

def _llm_answer(question, image_context, entities, facts, model_override):
    base_url, key = os.getenv("LLM_BASE_URL"), os.getenv("LLM_API_KEY")
    if not (base_url and key): return None
    model = model_override or os.getenv("LLM_MODEL")
    if not model: raise HTTPException(400, "Set LLM_MODEL or provide a model ID.")
    client = OpenAI(base_url=base_url, api_key=key, timeout=45)
    evidence = {"question":question,"image_analysis":image_context,"wikidata_entities":entities,"retrieved_facts":facts}
    result = client.chat.completions.create(model=model, messages=[{"role":"system","content":"You are an evidence-grounded research QA assistant. Use only supplied evidence, state when it is insufficient, and separate visual observations from retrieved facts."},{"role":"user","content":str(evidence)}], temperature=0.1)
    return result.choices[0].message.content or ""

@app.post("/api/qa")
def question_answer(request: QARequest):
    question = request.question.strip()
    terms = [t for t in re.sub(r"[^\w\s'-]", " ", question).split() if len(t)>2]
    search_term = " ".join(terms[-4:]) if terms else question
    entities, facts, query, retrieval_error = [], [], None, None
    try:
        entities = wikidata_search(search_term)
        if entities: query, facts = wikidata_facts(entities[0]["id"])
    except Exception as exc: retrieval_error = f"Wikidata retrieval unavailable: {type(exc).__name__}"
    try: answer = _llm_answer(question, request.image_context, entities, facts, request.model)
    except HTTPException: raise
    except Exception as exc:
        answer = None
        retrieval_error = (retrieval_error + "; " if retrieval_error else "") + f"LLM unavailable: {type(exc).__name__}"
    llm_used = answer is not None
    if answer is None:
        if facts:
            summary = "; ".join(f"{x['property']}: {x['value']}" for x in facts[:8])
            answer = f"Retrieved Wikidata facts for {entities[0]['label']}: {summary}. This is retrieval output, not an LLM-generated conclusion."
        elif entities: answer = f"Possible Wikidata match: {entities[0]['label']} — {entities[0]['description'] or 'no description available'}. No supporting facts were retrieved."
        else: answer = "Insufficient evidence retrieved. Try a more specific entity name or configure a supported LLM provider."
    return {"question":question,"answer":answer,"entities":entities,"facts":facts,"sparql_query":query,"retrieval_error":retrieval_error,"llm_used":llm_used}

if STATIC_DIR.exists():
    assets = STATIC_DIR / "assets"
    if assets.exists(): app.mount("/assets", StaticFiles(directory=assets), name="assets")
    @app.get("/{full_path:path}")
    def frontend(full_path: str):
        requested = (STATIC_DIR / full_path).resolve()
        if full_path and STATIC_DIR.resolve() in requested.parents and requested.is_file(): return FileResponse(requested)
        index = STATIC_DIR / "index.html"
        if index.exists(): return FileResponse(index)
        raise HTTPException(404, "Frontend build not found.")
