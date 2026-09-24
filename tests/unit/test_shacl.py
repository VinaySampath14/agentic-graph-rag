"""SHACL policy tests: review gaps warn; integrity errors fail."""
from pyshacl import validate
from rdflib import Graph, Namespace, RDF
from rdflib.namespace import SH


EX = Namespace("http://arxiv-cs.org/ontology#")
SHAPES = Graph().parse("ontology/shapes.ttl", format="turtle")


def _validate(data: Graph):
    return validate(
        data,
        shacl_graph=SHAPES,
        advanced=True,
        allow_warnings=True,
    )


def test_unclassified_method_is_warning_but_conforms():
    data = Graph()
    data.add((EX.UnreviewedMethod, RDF.type, EX.Method))
    conforms, report, _ = _validate(data)
    assert conforms is True
    assert (None, SH.resultSeverity, SH.Warning) in report


def test_single_reviewed_category_conforms():
    data = Graph()
    data.add((EX.LoRA, RDF.type, EX.Method))
    data.add((EX.LoRA, RDF.type, EX.FineTuningMethod))
    conforms, _, _ = _validate(data)
    assert conforms is True


def test_multiple_method_categories_are_hard_violation():
    data = Graph()
    data.add((EX.BadMethod, RDF.type, EX.Method))
    data.add((EX.BadMethod, RDF.type, EX.FineTuningMethod))
    data.add((EX.BadMethod, RDF.type, EX.AlignmentMethod))
    conforms, report, _ = _validate(data)
    assert conforms is False
    assert (None, SH.resultSeverity, SH.Violation) in report


def test_paper_without_author_is_hard_violation():
    data = Graph()
    data.add((EX.OrphanPaper, RDF.type, EX.Paper))
    conforms, report, _ = _validate(data)
    assert conforms is False
    assert (None, SH.resultSeverity, SH.Violation) in report
