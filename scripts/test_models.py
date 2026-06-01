from src.retrievers.models import RetrievalResult, GradeResult, GeneratorOutput
from src.retrievers.context_budget import apply_budget

r = RetrievalResult(context_text="test", source_type="vector")
print(r)

text, truncated = apply_budget(graph_context="graph data", vector_context="vector data")
print(text)
print("truncated:", truncated)
