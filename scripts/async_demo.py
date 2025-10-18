#!/usr/bin/env python3
from src.llm.orchestrator_async import Orchestrator, OpenAIAdapter, LocalFallbackAdapter, SafetyPolicy
import os

def run_demo():
    api_key = os.getenv("OPENAI_API_KEY")
    adapter = OpenAIAdapter(api_key) if api_key else LocalFallbackAdapter()
    orch = Orchestrator(adapter=adapter, policy=SafetyPolicy())
    prompt = "Summarize the following sensor snapshot and whether an alert is required."
    context = {"detections": [{"x": 10, "y": 20, "w": 30, "h": 40}], "eeg_len": 256}
    result = orch.safe_query(prompt, context)
    print("Result:")
    print(result)

if __name__ == "__main__":
    run_demo()
