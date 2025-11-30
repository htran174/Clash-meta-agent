#langgraph
from src.workflows.user_workflow import build_user_analytics_graph
from langgraph.server import serve_graph

graph = build_user_analytics_graph()

if __name__ == "__main__":
    serve_graph(graph, host="0.0.0.0", port=8123)
