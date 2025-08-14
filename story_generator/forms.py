from django import forms
from .models import StoryGeneration

class StoryPromptForm(forms.ModelForm):
    class Meta:
        model = StoryGeneration
        fields = ['user_prompt', 'audio_file']
        widgets = {
            'user_prompt': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Enter your creative prompt here...'
            }),
            'audio_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'audio/*'
            })
        }
    
    def clean(self):
        cleaned_data = super().clean()
        user_prompt = cleaned_data.get('user_prompt')
        audio_file = cleaned_data.get('audio_file')
        
        if not user_prompt and not audio_file:
            raise forms.ValidationError("Please provide either a text prompt or an audio file.")
        
        return cleaned_data