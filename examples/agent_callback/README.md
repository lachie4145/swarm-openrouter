# Agent Callback Example

This example demonstrates how to use the `llm_callback` and `llm_handle_callback` features in the `Agent` class to define custom logic for handling LLM interactions.

## Overview

In this example, we define a `Report Callback Agent` that uses custom callback functions to process instructions and handle responses. The agent is tasked with generating reports based on provided research.

## Files

- `agent_callback.py`: The main script demonstrating the use of custom callbacks in an agent.
- `README.md`: This file, providing an overview of the example.

## Running the Example

1. Ensure you have the necessary environment variables set, such as `OPENROUTER_API_KEY`.
2. Run the `agent_callback.py` script:

   ```bash
   python agent_callback.py
   ```

3. Observe the output, which will include the custom callback processing logs and the final response from the agent.

## Key Concepts

- **Custom LLM Callback**: A function that processes the agent's instructions and simulates a response from an external LLM.
- **Custom LLM Handle Callback**: A function that processes the response from the LLM callback for further customization.