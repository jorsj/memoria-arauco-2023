import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part

vertexai.init(project="sandcastle-401718", location="us-central1")

model = GenerativeModel("gemini-1.5-flash-002")

with open("./prompt_template.txt", "r") as file:
    prompt_template = file.read()

with open("./system_instructions.txt", "r") as file:
    system_instruction = file.read()

document = Part.from_uri(
    mime_type="text/markdown",
    uri="gs://arauco-sandcastle-401718/document.md",
)

model = GenerativeModel("gemini-1.5-flash-002", system_instruction=[system_instruction])

def response_generator():
    gemini_prompt = prompt_template.format(
        messages=st.session_state.messages,
        question=prompt
    )
    responses = model.generate_content(
        [gemini_prompt, document],
        stream=True
    )
    for response in responses:
        yield response.text


if __name__ == "__main__":
    st.title("Memoria Arauco 2023 🌳")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hola, ¿en qué te puedo ayudar?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator)
        st.session_state.messages.append({"role": "assistant", "content": response})
