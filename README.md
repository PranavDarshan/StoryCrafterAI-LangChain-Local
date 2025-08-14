# StoryCrafterAI-LangChain-Local

StoryCrafterAI is a creative storytelling application that leverages the power of large language models (LLMs) and AI image generation to bring your ideas to life. This project is designed to run completely locally on your CPU, ensuring your privacy and creative control.

## Features

*   **AI-Powered Story Generation:**  Uses LangChain to generate unique and engaging stories based on your prompts.
*   **Local Image Generation:** Creates custom images for your stories using a locally run Stable Diffusion model.
*   **Audio-to-Text:**  Transcribe your story ideas from audio recordings.
*   **Web-Based Interface:**  An easy-to-use interface built with Django.

## How It Works

1.  **Provide a Prompt:** You can either type a story prompt or upload an audio file.
2.  **Story Generation:** The application uses a local LLM via LangChain to generate a story based on your prompt.
3.  **Image Generation:**  The application generates a character and a background image using local Stable Diffusion. These images are then combined.
4.  **View Your Story:** The final story and image are displayed in the browser.

## Models Used

This project uses the following models, all running locally:

*   **Story Generation:**
    *   **Primary:** `gpt2`
    *   **Fallback:** `distilgpt2`
*   **Audio Transcription:** `whisper-tiny`
*   **Image Generation:** `nota-ai/bk-sdm-small`
    *   This is a Block-removed Knowledge-distilled Stable Diffusion Model (BK-SDM), which is an architecturally compressed SDM for efficient general-purpose text-to-image synthesis. This model is built by (i) removing several residual and attention blocks from the U-Net of Stable Diffusion v1.4 and (ii) distillation pretraining on only 0.22M LAION pairs (fewer than 0.1% of the full training set). Despite being trained with very limited resources, this compact model can imitate the original SDM by benefiting from transferred knowledge.

## Local and Private

This project is configured to run all AI models (LLMs and Stable Diffusion) locally on your machine's CPU. This means:

*   **No API Keys Needed:** You don't need to sign up for any paid services.
*   **Privacy:** Your data and creations are never sent to the cloud.
*   **Full Control:** You have complete control over the models and the creative process.

## Getting Started

### Prerequisites

*   Python 3.10+
*   A C++ compiler (required for some of the dependencies)
*   Redis

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd StoryCrafterAI-LangChain-Local
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the database migrations:**
    ```bash
    python manage.py migrate
    ```

5.  **Start the development server:**
    ```bash
    python manage.py runserver
    ```

6.  **Open your browser:** Navigate to `http://127.0.0.1:8000/` to start creating stories.

## Project Structure

*   `creative_app/`: The main Django project directory.
*   `story_generator/`: The core application for story generation.
*   `static/`: Static files (CSS, JavaScript, images).
*   `media/`: User-uploaded files.
*   `generated_images/`: AI-generated images.
*   `manage.py`: The Django management script.
*   `requirements.txt`: The list of Python dependencies.
*   `db.sqlite3`: The SQLite database file.