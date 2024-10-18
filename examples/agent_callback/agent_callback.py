from swarm import Swarm, Agent
import os

# Define a custom LLM callback function
def custom_llm_callback(instructions):
    print(f"Custom LLM Callback received instructions: {instructions}")
    # Simulate a response from an external LLM
    return "This is a simulated response from a custom LLM."

# Define a custom LLM handle callback function
def custom_llm_handle_callback(response):
    print(f"Custom LLM Handle Callback received response: {response}")
    # Process the response further if needed
    return f"Processed response: {response}"

# Define a function that the agent can call
def generate_report():
    return "Report generated successfully."

# Create an agent with the custom callbacks
report_callback_agent = Agent(
    name="Report Callback Agent",
    instructions="You are a report writing agent. Your task is to create detailed reports based on the research provided.",
    llm_callback=custom_llm_callback,
    llm_handle_callback=custom_llm_handle_callback,
    functions=[generate_report]
)

# Initialize Swarm with custom configurations
client = Swarm(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    default_model="google/gemini-flash-1.5"
)

# Run the agent
response = client.run(
    agent=report_callback_agent,
    messages=[{"role": "user", "content": "Please generate a report."}],
)

# Print the final response
print(response.messages[-1]["content"])