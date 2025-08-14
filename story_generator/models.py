from django.db import models
import os

class StoryGeneration(models.Model):
    user_prompt = models.TextField()
    story = models.TextField(blank=True)
    character_description = models.TextField(blank=True)
    background_description = models.TextField(blank=True)
    character_image_prompt = models.TextField(blank=True)
    background_image_prompt = models.TextField(blank=True)
    combined_image = models.ImageField(upload_to='generated_images/', blank=True)
    audio_file = models.FileField(upload_to='audio_uploads/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def delete(self, *args, **kwargs):
        # Clean up files when deleting model instance
        if self.combined_image:
            if os.path.isfile(self.combined_image.path):
                os.remove(self.combined_image.path)
        if self.audio_file:
            if os.path.isfile(self.audio_file.path):
                os.remove(self.audio_file.path)
        super().delete(*args, **kwargs)