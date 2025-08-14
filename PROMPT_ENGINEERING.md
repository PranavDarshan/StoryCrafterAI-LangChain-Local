# Prompt Engineering

This document outlines the prompts used in the Creative Story Generator project. These prompts are designed to generate a story, character descriptions, and background descriptions based on a user's initial input.

## Story Generation

The following prompt is used to generate the main story.

### Story Prompt

```
Write a creative short story (2-3 paragraphs) based on this prompt:

Prompt: {user_prompt}
Create a clean story which is also used for langchain chaining
Story:
```

-   `{user_prompt}`: This is the initial prompt provided by the user.

## Character Description

This prompt is used to generate a detailed description of the main character based on the generated story.

### Character Description Prompt

```
Based on this story, describe the main character in detail:

Story: {story}

Describe the character's:
- Physical appearance
- Clothing and style
- Facial expression
- Pose and demeanor

Character description:
```

-   `{story}`: This is the story generated from the previous step.

## Background Description

This prompt is used to generate a description of the story's setting and background.

### Background Description Prompt

```
Based on this story, describe the setting and background:

Story: {story}

Describe:
- The location and environment
- Time of day and lighting
- Atmosphere and mood
- Visual details of the scene

Background description:
```

-   `{story}`: This is the story generated from the first step.

## Image Generation Prompts

The character and background descriptions are then used to generate prompts for an image generation model (e.g., Stable Diffusion). The prompts are optimized with additional keywords to improve image quality.

### Image Prompt Optimization

The following logic is used to create the image prompts:

-   **Character Prompt:** `{character_description}, portrait, detailed, high quality, digital art, fantasy style, concept art`
-   **Background Prompt:** `{background_description}, landscape, detailed, high quality, digital art, fantasy style, matte painting`

The prompts are then cleaned by removing certain keywords and adding others to enhance the final image.
