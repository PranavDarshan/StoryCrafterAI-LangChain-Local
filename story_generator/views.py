from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
import logging
import uuid
from .forms import StoryPromptForm
from .models import StoryGeneration
from .langchain_service import StoryGenerationService
from .image_service import ImageGenerationService
from .audio_service import AudioService

logger = logging.getLogger(__name__)

def home(request):
    """Home page with form"""
    if request.method == 'POST':
        form = StoryPromptForm(request.POST, request.FILES)
        if form.is_valid():
            return process_generation(request, form)
    else:
        form = StoryPromptForm()
    
    return render(request, 'story_generator/home.html', {'form': form})

def process_generation(request, form):
    """Process the story generation"""
    try:
        # Save form data
        story_gen = form.save()
        
        # Get user prompt
        user_prompt = story_gen.user_prompt
        
        # Handle audio transcription if provided
        if story_gen.audio_file:
            audio_service = AudioService()
            transcription = audio_service.transcribe_audio(story_gen.audio_file)
            if transcription:
                user_prompt = transcription
                story_gen.user_prompt = transcription
            else:
                messages.error(request, "Failed to transcribe audio. Please try again.")
                return redirect('home')
        
        # Initialize services
        langchain_service = StoryGenerationService()
        image_service = ImageGenerationService()
        
        # Generate story and descriptions
        logger.info("Generating story and descriptions...")
        content = langchain_service.generate_story_and_descriptions(user_prompt)
        
        # Update model with generated content
        story_gen.story = content['story']
        story_gen.character_description = content['character_description']
        story_gen.background_description = content['background_description']
        
        # Create image prompts
        image_prompts = langchain_service.create_image_prompts(
            content['character_description'],
            content['background_description']
        )
        
        story_gen.character_image_prompt = image_prompts['character_prompt']
        story_gen.background_image_prompt = image_prompts['background_prompt']
        
        # Generate images
        logger.info("Generating character image...")
        character_image = image_service.generate_character_image(
            image_prompts['character_prompt']
        )
        
        logger.info("Generating background image...")
        background_image = image_service.generate_background_image(
            image_prompts['background_prompt']
        )
        
        # Combine images
        logger.info("Combining images...")
        combined_image = image_service.combine_images(character_image, background_image)
        
        # Save combined image
        filename = f"combined_{uuid.uuid4().hex}.jpg"
        image_path = image_service.save_image(combined_image, filename)
        
        if image_path:
            story_gen.combined_image = image_path
        
        story_gen.save()
        
        messages.success(request, "Story and images generated successfully!")
        return render(request, 'story_generator/result.html', {'story_gen': story_gen})
        
    except Exception as e:
        logger.error(f"Error in process_generation: {e}")
        messages.error(request, f"An error occurred: {str(e)}")
        return redirect('home')

def result_view(request, pk):
    """View individual result"""
    try:
        story_gen = StoryGeneration.objects.get(pk=pk)
        return render(request, 'story_generator/result.html', {'story_gen': story_gen})
    except StoryGeneration.DoesNotExist:
        messages.error(request, "Story not found.")
        return redirect('home')