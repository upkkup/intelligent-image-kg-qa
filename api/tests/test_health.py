from fastapi.testclient import TestClient
from app.main import app, _scene_relations

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_short_question_rejected():
    response = client.post("/api/qa", json={"question": "x"})
    assert response.status_code == 422


def test_image_endpoint_rejects_unsupported_media_type():
    response = client.post(
        "/api/analyze-image",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_scene_relations_are_returned_as_structured_triples():
    detections = [
        {"label": "person", "bbox": [0, 0, 10, 20]},
        {"label": "cup", "bbox": [30, 0, 40, 10]},
    ]
    relations = _scene_relations(detections)
    assert len(relations) == 1
    assert relations[0]["subject"] == "person"
    assert relations[0]["object"] == "cup"
    assert relations[0]["relation"] in {"left of", "right of", "above", "below", "contains/overlaps"}


def test_qa_returns_retrieval_facts_without_external_network(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "wikidata_search", lambda term: [
        {"id": "Q142", "label": "France", "description": "country in Europe"}
    ])
    monkeypatch.setattr(main, "wikidata_facts", lambda entity_id: (
        "SELECT ?propertyLabel ?valueLabel WHERE { }",
        [{"property": "capital", "value": "Paris"}],
    ))
    monkeypatch.setattr(main, "_llm_answer", lambda *args: None)

    response = client.post("/api/qa", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["entities"][0]["id"] == "Q142"
    assert payload["facts"] == [{"property": "capital", "value": "Paris"}]
    assert "Paris" in payload["answer"]
    assert payload["llm_used"] is False


def test_question_search_candidates_remove_question_fillers():
    import app.main as main

    candidates = main._candidate_search_terms("What is the capital of France?")
    assert "France" in candidates
    assert "capital France" in candidates
    assert all("what is the capital" != candidate.casefold() for candidate in candidates)


def test_wikidata_facts_query_includes_literal_values(monkeypatch):
    import app.main as main

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": {"bindings": [
                {"propertyLabel": {"value": "population"}, "valueLabel": {"value": "68000000"}}
            ]}}

    captured = {}
    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs["params"]
        return FakeResponse()

    monkeypatch.setattr(main.requests, "get", fake_get)
    query, facts = main.wikidata_facts("Q142")
    assert "wd:Q142" in query
    assert "wikibase:directClaim" in query
    assert "FILTER(isIRI" not in query
    assert facts[0]["value"] == "68000000"


def test_wikidata_facts_rejects_invalid_entity_id():
    import pytest
    import app.main as main

    with pytest.raises(ValueError):
        main.wikidata_facts("not-an-entity")
