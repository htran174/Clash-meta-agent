from src.workflows.user_workflow import build_user_analytics_graph


def main():
    graph = build_user_analytics_graph()

    state = graph.invoke({"player_tag": "8C8JJQLG"})  # or another test tag

    print("\n--- STATE KEYS ---")
    print(state.keys())

    print("\n--- RAW BATTLES COUNT ---")
    print(len(state.get("battles_raw", [])))

    print("\n--- FILTERED BATTLES COUNT ---")
    print(len(state.get("battles_filtered", [])))

    print("\n--- ANALYTICS SUMMARY ---")
    summary = state.get("user_analytics", {}).get("summary", {})
    print(summary)

    print("\n--- NOTES ---")
    for n in state.get("notes", []):
        print("-", n)


if __name__ == "__main__":
    main()
