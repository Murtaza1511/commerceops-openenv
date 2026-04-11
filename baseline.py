from app.baseline_runner import run_all_tasks_http

BASE_URL = "http://127.0.0.1:7860"


def run_baseline():
    summary = run_all_tasks_http(BASE_URL, use_openai=False)

    print("\n🚀 Running Baseline Across All Tasks\n")
    for item in summary["results"]:
        print(f"🔹 Running {item['task_id']} ({item['difficulty']})")
        for index, reward in enumerate(item["rewards"], start=1):
            print(f"Step {index} | Reward: {reward:.2f}")
        print(f"✅ Score: {item['score']:.2f}\n")

    print("📊 FINAL RESULTS")
    print("-------------------")
    for item in summary["results"]:
        print(f"{item['task_id']}: {item['score']:.2f}")
    print(f"\n🏆 Average Score: {summary['average_score']:.2f}")


if __name__ == "__main__":
    run_baseline()
