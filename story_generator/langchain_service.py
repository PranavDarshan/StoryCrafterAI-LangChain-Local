import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StoryGenerationService:
    def __init__(self):
        self.device = "cpu"
        logger.info("Initializing LangChain Story Generation Service with local models")
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the language model for CPU usage"""
        try:
            # Use GPT-2 small for reliable CPU performance
            model_name = "gpt2"
            
            logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer and model
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu"
            )
            
            # Set pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create text generation pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,  # Limit output length
                temperature=0.8,
                do_sample=True,
                device=-1,  # CPU
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Create LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("Model initialized successfully on CPU")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Fallback initialization if main model fails"""
        try:
            # Use the smallest possible model
            pipe = pipeline(
                "text-generation",
                model="distilgpt2",
                max_new_tokens=150,
                temperature=0.7,
                device=-1,
                return_full_text=False
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("Fallback model (DistilGPT2) initialized")
        except Exception as e:
            logger.error(f"Fallback model also failed: {e}")
            # Create a mock LLM for testing
            self.llm = self._create_mock_llm()
    
    def _create_mock_llm(self):
        """Create a mock LLM for testing purposes"""
        class MockLLM:
            def __call__(self, prompt):
                return f"Generated story based on: {prompt}"
            
            def __getattr__(self, name):
                return lambda *args, **kwargs: f"Generated content for {name}"
        
        return MockLLM()
    
    def generate_story_and_descriptions(self, user_prompt):
        """Generate story with character and background descriptions using real AI"""
        try:
            logger.info("Starting story generation...")

            
            # Generate story
            story = self._generate_story(user_prompt)
    
            logger.info("Story generated")
            
            # Generate character description
            character_desc = self._generate_character_description(story, user_prompt)
            logger.info("Character description generated")
            
            # Generate background description  
            background_desc = self._generate_background_description(story, user_prompt)
            logger.info("Background description generated")
            
            return {
                'story': story,
                'character_description': character_desc,
                'background_description': background_desc
            }
            
        except Exception as e:
            logger.error(f"Error in story generation: {e}")
            return self._generate_enhanced_fallback(user_prompt)
    
    def _generate_story(self, user_prompt):
        """Generate story using LangChain"""
        story_template = """Write a creative short story (2-3 paragraphs) based on this prompt:
        
Prompt: {user_prompt}
Create a clean story which is also used for langchain chaining
Story:"""
        
        try:
            prompt = PromptTemplate(
                input_variables=["user_prompt"],
                template=story_template
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = chain.run(user_prompt=user_prompt)
            
            # Clean up the result
            print(result)
            story = result.strip()
            if len(story) < 50:  # If too short, enhance it
                story = self._enhance_short_story(story, user_prompt)
            
            return story
            
        except Exception as e:
            logger.error(f"Error generating story: {e}")
            return self._fallback_story(user_prompt)
    
    def _generate_character_description(self, story, user_prompt):
        """Generate character description"""
        char_template = """Based on this story, describe the main character in detail:

Story: {story}

Describe the character's:
- Physical appearance
- Clothing and style  
- Facial expression
- Pose and demeanor

Character description:"""
        
        try:
            prompt = PromptTemplate(
                input_variables=["story"],
                template=char_template
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = chain.run(story=story[:500])  # Limit input length
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error generating character description: {e}")
            return self._fallback_character(user_prompt)
    
    def _generate_background_description(self, story, user_prompt):
        """Generate background description"""
        bg_template = """Based on this story, describe the setting and background:

Story: {story}

Describe:
- The location and environment
- Time of day and lighting
- Atmosphere and mood
- Visual details of the scene

Background description:"""
        
        try:
            prompt = PromptTemplate(
                input_variables=["story"],
                template=bg_template
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = chain.run(story=story[:500])
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error generating background description: {e}")
            return self._fallback_background(user_prompt)
    
    def _enhance_short_story(self, short_story, user_prompt):
        """Enhance a short story with more content"""
        enhancement = f" The adventure of {user_prompt} continued as our protagonist discovered new challenges and opportunities. With courage and determination, they faced each obstacle, learning valuable lessons that would shape their destiny. The journey ahead promised even greater wonders and the chance to make a lasting impact on the world around them."
        return short_story + enhancement
    
    def _fallback_story(self, user_prompt):
        """Fallback story generation"""
        return f"In a realm where {user_prompt} holds great significance, an epic tale unfolds. Our brave protagonist embarks on a journey filled with wonder, facing challenges that will test their courage and reveal their true character. Through perseverance and wisdom, they discover that the greatest adventures often begin with a single step into the unknown, and that destiny awaits those bold enough to pursue their dreams."
    
    def _fallback_character(self, user_prompt):
        """Fallback character description"""
        return f"A noble hero with determined eyes and a courageous spirit, inspired by the essence of {user_prompt}. They wear practical yet elegant attire suitable for adventure, with a confident posture that speaks of inner strength. Their expression shows both wisdom gained from experience and hope for the journey ahead."
    
    def _fallback_background(self, user_prompt):
        """Fallback background description"""  
        return f"A magnificent landscape that embodies the spirit of {user_prompt}, with sweeping vistas and magical elements. The scene is illuminated by warm, golden light that creates an atmosphere of hope and possibility. Ancient trees, rolling hills, and distant mountains form a backdrop perfect for epic adventures and legendary tales."
    
    def _generate_enhanced_fallback(self, user_prompt):
        """Enhanced fallback with more sophisticated templates"""
        return {
            'story': self._fallback_story(user_prompt),
            'character_description': self._fallback_character(user_prompt),
            'background_description': self._fallback_background(user_prompt)
        }
    
    def create_image_prompts(self, character_desc, background_desc):
        """Create optimized prompts for Stable Diffusion"""
        # Optimize character prompt for image generation
        character_prompt = f"{character_desc}, portrait, detailed, high quality, digital art, fantasy style, concept art"
        
        # Optimize background prompt  
        background_prompt = f"{background_desc}, landscape, detailed, high quality, digital art, fantasy style, matte painting"
        
        # Clean and limit prompts
        character_prompt = self._clean_prompt(character_prompt)[:200]
        background_prompt = self._clean_prompt(background_prompt)[:200]
        
        return {
            'character_prompt': character_prompt,
            'background_prompt': background_prompt
        }
    
    def _clean_prompt(self, prompt):
        """Clean prompt for image generation"""
        # Remove problematic words/phrases
        removals = ['describe', 'description:', 'character:', 'background:', 'story:', '\n']
        cleaned = prompt.lower()
        for removal in removals:
            cleaned = cleaned.replace(removal, '')
        
        # Add positive keywords
        if 'portrait' not in cleaned and 'landscape' not in cleaned:
            cleaned = f"beautiful {cleaned}"
        
        return cleaned.strip()