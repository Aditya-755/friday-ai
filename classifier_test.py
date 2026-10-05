from classifier import classify_both

TEST_CASES = [
    ("hey friday", "greeting", "idle", "casual_chat"),
    ("good morning", "greeting", "idle", "casual_chat"),
    ("what time is it", "time", "idle", "idle"),
    ("what's today's date", "date", "idle", "idle"),
    ("bye friday", "exit", "idle", "idle"),
    (" me a story", "general", "idle", "storytelling"),
    ("continue the story", "general", "storytelling", "storytelling"),
    ("make the ending happier", "general", "storytelling", "storytelling"),
    ("help me fix my Python code", "general", "idle", "coding"),
    ("why is my program crashing", "general", "coding", "coding"),
    ("help me plan my career", "general", "idle", "planning"),
    ("what should I learn next", "general", "planning", "planning"),
    ("what is machine learning", "general", "idle", "question_answering"),
    ("how does a CPU work", "general", "idle", "question_answering"),
    ("do you ever get bored", "general", "casual_chat", "casual_chat"),

    # State switches
    ("now help me debug my code", "general", "storytelling", "coding"),
    ("forget the code, tell me a story", "general", "coding", "storytelling"),
    ("forget the story, help me plan my career", "general", "storytelling", "planning"),
]

def run_tests():
    intent_correct = 0
    state_correct = 0

    for text, expected_intent, previous_state, expected_state in TEST_CASES:
        intent, intent_conf, state, state_conf = classify_both(
            text,
            previous_state
        )

        intent_ok = "✓" if intent == expected_intent else "✗"
        state_ok = "✓" if state == expected_state else "✗"

        if intent == expected_intent:
            intent_correct += 1

        if state == expected_state:
            state_correct += 1

        print(f'\nInput: "{text}"')
        print(f"  Intent: {intent_ok} {intent} | expected: {expected_intent} | confidence: {intent_conf}")
        print(f"  State : {state_ok} {state} | previous: {previous_state} | expected: {expected_state} | confidence: {state_conf}")

    total = len(TEST_CASES)

    print("\n" + "=" * 50)
    print(f"Intent accuracy: {intent_correct}/{total} ({100 * intent_correct / total:.1f}%)")
    print(f"State accuracy : {state_correct}/{total} ({100 * state_correct / total:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    run_tests()