import streamlit as st
import time
from google.api_core.exceptions import ResourceExhausted
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

vertexai.init(project="sandcastle-401718", location="us-central1")

with open("./prompt_template.txt", "r") as file:
    prompt_template = file.read()

with open("./system_instructions.txt", "r") as file:
    system_instruction = file.read()

with open("./welcome.txt", "r") as file:
    welcome = file.read()
    
with open("./error.txt", "r") as file:
    error = file.read()

document = Part.from_uri(
    mime_type="text/markdown",
    uri="gs://arauco-sandcastle-401718/document.md",
)

model = GenerativeModel("gemini-1.5-flash-002", system_instruction=[system_instruction])


def response_generator(max_retries=5, initial_delay=1):
    """
    Generates responses from a language model with exponential backoff retry for ResourceExhausted exceptions.

    Args:
        prompt_template: The template for generating prompts.
        model: The language model object.
        document: The document to be used in the prompt.
        prompt: The user's prompt.
        max_retries: The maximum number of retries.
        initial_delay: The initial delay in seconds.

    Yields:
        str: Chunks of the generated response.
    """

    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        gemini_prompt = prompt_template.format(
            messages=st.session_state.messages,
            question=prompt
        )

        try:
            responses = model.generate_content(
                [document, gemini_prompt],
                generation_config=GenerationConfig(
                    temperature=0
                ),
                stream=True
            )
            for response in responses:
                yield response.text
            return  # Exit the function if successful

        except ResourceExhausted as e:
            retries += 1
            if retries > max_retries:
                print(f"Giving up after {max_retries} retries due to ResourceExhausted: {e}")
                yield error
                return  # Re-raise the exception after max retries
            else:
                print(f"ResourceExhausted: Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff


if __name__ == "__main__":
    st.title("Reporte Integrado Arauco 2023 🌳")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": welcome
            }
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator)
        st.session_state.messages.append({"role": "assistant", "content": response})
