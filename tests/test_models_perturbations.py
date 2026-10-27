from dataclasses import asdict

import pytest

from hinglish_rag_probe import Document, Query, QueryVariant
from hinglish_rag_probe.perturbations import TRANSFORMS, add_variant, perturb_query


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id": ""},
        {"id": "q"},
        {"id": "q", "english": ""},
        {"id": "q", "english": "hi", "relevant_ids": "doc"},
        {"id": "q", "english": "hi", "relevance": {"d": -1}},
        {"id": "q", "english": "hi", "relevant_ids": {"d"}, "relevance": {"x": 1}},
        {"id": "q", "english": "hi", "human_validated": "false"},
        {"id": "q", "english": "hi", "variants": (QueryVariant("english", "hello"),)},
    ],
)
def test_query_validation(kwargs):
    with pytest.raises(ValueError):
        Query(**kwargs)


def test_document_and_variant_validation():
    with pytest.raises(ValueError):
        Document("a", "")
    with pytest.raises(ValueError):
        QueryVariant("hindi", "hi", "false")
    assert Query("q", hindi="पानी", relevance={"d": 2}).judgments() == {"d": 2}
    assert Query("q", english="water").judgments() is None


@pytest.mark.parametrize("transformation", TRANSFORMS)
def test_perturb_reproducibility_and_provenance(transformation):
    query = Query(
        "q", english="Where does water come from?", relevant_ids={"d"}, human_validated=True
    )
    a = perturb_query(query, transformation=transformation, seed=12, intensity=0.4)
    b = perturb_query(query, transformation=transformation, seed=12, intensity=0.4)
    assert asdict(a) == asdict(b)
    assert a.relevant_ids == query.relevant_ids
    assert a.variants[0].human_validated is False
    assert a.variants[0].settings["original_text"] == query.english
    assert a.variants[0].settings["original_query_id"] == "q"
    assert not query.variants
    assert (
        perturb_query(query, transformation=transformation, intensity=0).variants[0].text
        == query.english
    )


def test_added_variants_and_bad_transforms():
    q = Query("q", english="water")
    result = add_variant(
        q, condition="romanized_hindi", text="paani", kind="transliteration", human_validated=True
    )
    assert result.variants[0].human_validated
    for kwargs in (
        {"transformation": "translate"},
        {"intensity": -1},
        {"source_condition": "hindi"},
        {"alphabet": ""},
    ):
        with pytest.raises(ValueError):
            perturb_query(q, **kwargs)
    with pytest.raises(ValueError):
        add_variant(q, condition="x", text="hi", kind="wrong")
    assert (
        perturb_query(Query("x", english="a"), transformation="delete", intensity=1)
        .variants[0]
        .text
        == "a"
    )
