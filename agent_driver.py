import pprint
from langchain_core.messages import message_to_dict, messages_to_dict, BaseMessage
from tools_sql_agent import run_agent, run_agent_with_history


user_prompts = [
    "which category has the 4th largest expense"
]
user_q = "What is the total amount for this category"
updated, answer = run_agent_with_history(
    history=user_prompts,
    user_text=user_q,
)

print("----------------------------------------   :Updated:   --------------------------------------------------")
if isinstance(updated, list):
    for msg in updated:
        pprint.pprint(message_to_dict(msg) if isinstance(msg, BaseMessage) else msg)
else:
    pprint.pprint(updated)
print("--------------------------------------------------------------------------------------------------")


print("----------------------------------------   :Answer:  --------------------------------------------------")
pprint.pprint(answer)