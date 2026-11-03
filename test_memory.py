from memory.database import (
    initialize_database,
    start_conversation,
    save_message
)

initialize_database()

session = start_conversation()

save_message(session, "user", "Hello FRIDAY")

save_message(
    session,
    "friday",
    "Hey Boss! What's up?"
)

print("Memory test completed!")