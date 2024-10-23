import streamlit as st
import time
import json
import datetime
from google.api_core.exceptions import ResourceExhausted, InvalidArgument
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, GenerationConfig
from vertexai.preview import caching
import requests
import google.auth.transport.requests
from google.auth import default

BUCKET_NAME = "arauco-sandcastle-401718"
BLOB_NAME = "document.md"
PROJECT_ID = "sandcastle-401718"
LOCATION = "us-central1"
CACHE_NAME = "reporte-arauco-2023"

with open("prompt_template.txt", "r") as file:
    prompt_template = file.read()

with open("welcome.txt", "r") as file:
    welcome = file.read()

with open("error.txt", "r") as file:
    error = file.read()


def create_context_cache():
    """
    Creates a Context Cache with system instructions and initial content.

    Returns:
        A vertexai.preview.caching.CachedContent object.
    """
    try:
        with open("./system_instructions.txt", "r") as system_instructions_file:
            system_instruction = system_instructions_file.read()
        print("Successfully loaded system instructions.")
    except FileNotFoundError:
        print("system_instructions.txt not found.", exc_info=True)
        raise  # Re-raise the exception to halt execution
    try:
        document = Part.from_uri(
            f"gs://{BUCKET_NAME}/{BLOB_NAME}",  # Added bucket and blob name
            mime_type="text/markdown",
        )
        print(f"Loaded content from gs://{BUCKET_NAME}/{BLOB_NAME}")
    except Exception as e:
        print(f"Error loading content from GCS: {e}", exc_info=True)
        raise
    try:
        cache = caching.CachedContent.create(
            model_name="gemini-1.5-flash-002",
            system_instruction=system_instruction,
            contents=document,
            ttl=datetime.timedelta(days=360),
            display_name=CACHE_NAME,
        )
        return cache
    except Exception as e:
        print(f"Error creating context cache: {e}", exc_info=True)
        raise


def refresh_cached_context():
    """
    Refreshes the cached context and generates a new model instance.

    This function attempts to fetch the cached context. If the fetch fails, it
    creates a new context cache. It then returns the cached content and a new
    GenerativeModel instance initialized with the cached content.

    Returns:
        A tuple containing the cached content and a new GenerativeModel instance.

    Raises:
        Exception: If an error occurs while fetching or creating the cached context.
    """
    try:
        cached_content = fetch_cached_content()
    except Exception as e:
        print(f"Creating new context cache because of error: {str(e)}")
        cached_content = create_context_cache()
    return cached_content, GenerativeModel.from_cached_content(cached_content)


def fetch_cached_content():
    """
    Retrieves cached content from Vertex AI.

    This function fetches cached content from Vertex AI. It first authenticates using
    default credentials, then builds the request URL and headers.

    Args:
        None: This function takes no parameters.

    Returns:
        caching.CachedContent: If a cached content with the specified name is found.
        None: If no cached content with the specified name is found.

    Raises:
        Exception: If an error occurs while fetching or parsing the cached content.
    """
    creds, _ = default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/cachedContents"
    headers = {"Authorization": f"Bearer {creds.token}"}
    response = requests.request("GET", url, headers=headers).json()
    try:
        response = requests.request("GET", url, headers=headers).json()
        for cached_content in response["cachedContents"]:
            if cached_content["displayName"] == CACHE_NAME:
                print(f"Found context cache with name {cached_content["name"]}")
                return caching.CachedContent(cached_content_name=cached_content["name"])

        raise Exception

    except Exception:
        print("No cached content found.")
        raise  # Re-raise the exception to propagate the error


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
    print(json.dumps(st.session_state.messages, ensure_ascii=False))
    while retries <= max_retries:
        gemini_prompt = prompt_template.format(
            messages=st.session_state.messages, question=prompt
        )

        try:
            responses = st.model.generate_content(
                [gemini_prompt],
                generation_config=GenerationConfig(temperature=0),
                stream=True,
            )
            for response in responses:
                yield response.text
            return  # Exit the function if successful

        except ResourceExhausted as e:
            retries += 1
            if retries > max_retries:
                print(
                    f"Giving up after {max_retries} retries due to ResourceExhausted: {e}"
                )
                yield error
                return  # Exit the function after max retries
            else:
                print(f"ResourceExhausted: Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
        except InvalidArgument as e:
            print(f"Error querying the context cache: {str(e)}")
            st.cached_content, st.model = refresh_cached_context()


if __name__ == "__main__":
    if "initialized" not in st.session_state or not st.session_state.initialized:
        print("Starting application...")
        st.set_page_config(page_title="Reporte Integrado Arauco 2023", page_icon="🌳")
        vertexai.init(project="sandcastle-401718", location="us-central1")
        st.cached_content, st.model = refresh_cached_context()
        st.session_state.initialized = True

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": welcome}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator)
        st.session_state.messages.append({"role": "assistant", "content": response})
