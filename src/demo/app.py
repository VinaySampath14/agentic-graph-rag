import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Agentic Graph RAG
        **Coming soon** — self-correcting context engine over 2,000 arXiv CS papers.

        Routes queries between vector, graph, and community retrieval modes.
        Rewrites on failure. Explains every decision.

        Built on Neo4j · Qdrant · LangGraph · Groq LLaMA 3.3 70B

        [GitHub](https://github.com/VinaySampath14/agentic-graph-rag)
        """
    )

demo.launch()
