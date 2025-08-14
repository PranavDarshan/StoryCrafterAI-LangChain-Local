import logging
import os
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import cv2
import numpy as np
from django.conf import settings
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageGenerationService:
    def __init__(self):
        self.device = "cpu"
        logger.info("Initializing Stable Diffusion for local image generation")
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize Stable Diffusion pipeline for CPU"""
        try:
            # Use Stable Diffusion 1.4 (smaller and more CPU-friendly)
            model_id = "CompVis/stable-diffusion-v1-4"
            
            logger.info("Loading Stable Diffusion model...")
            
            # Initialize with CPU-optimized settings
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,  # Use float32 for CPU
                safety_checker=None,  # Disable for speed
                requires_safety_checker=False,
                use_auth_token=False
            )
            
            # Move to CPU and optimize
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()  # Reduce memory usage
            self.pipe.enable_cpu_offload()  # CPU optimization
            
            logger.info("Stable Diffusion pipeline initialized successfully on CPU")
            
        except Exception as e:
            logger.error(f"Error initializing Stable Diffusion: {e}")
            logger.info("Attempting to load smaller model...")
            self._initialize_tiny_model()
    
    def _initialize_tiny_model(self):
        """Initialize a smaller model for CPU"""
        try:
            # Try a smaller, faster model
            model_id = "nota-ai/bk-sdm-small"
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()
            
            logger.info("Smaller Stable Diffusion model initialized")
            
        except Exception as e:
            logger.error(f"Could not load any Stable Diffusion model: {e}")
            self.pipe = None
    
    def generate_character_image(self, prompt, width=512, height=512):
        """Generate character image using Stable Diffusion"""
        if not self.pipe:
            logger.warning("No model available, creating placeholder")
            return self._create_placeholder_image(width, height, "Character")
        
        try:
            logger.info(f"Generating character image with prompt: {prompt[:100]}...")
            
            # CPU-optimized generation parameters
            image = self.pipe(
                prompt,
                num_inference_steps=20,  # Reduced for CPU speed
                width=width,
                height=height,
                guidance_scale=7.5,
                negative_prompt="ugly, blurry, low quality, distorted"
            ).images[0]
            
            logger.info("Character image generated successfully")
            return image
            
        except Exception as e:
            logger.error(f"Error generating character image: {e}")
            return self._create_placeholder_image(width, height, "Character")
    
    def generate_background_image(self, prompt, width=512, height=512):
        """Generate background image using Stable Diffusion"""
        if not self.pipe:
            logger.warning("No model available, creating placeholder")
            return self._create_placeholder_image(width, height, "Background")
        
        try:
            logger.info(f"Generating background image with prompt: {prompt[:100]}...")
            
            # CPU-optimized generation
            image = self.pipe(
                prompt,
                num_inference_steps=20,  # Reduced for CPU
                width=width,
                height=height,
                guidance_scale=7.5,
                negative_prompt="ugly, blurry, low quality, people, characters"
            ).images[0]
            
            logger.info("Background image generated successfully")
            return image
            
        except Exception as e:
            logger.error(f"Error generating background image: {e}")
            return self._create_placeholder_image(width, height, "Background")
    
    def combine_images(self, character_img, background_img):
        """Combine character and background images using advanced blending"""
        try:
            # Resize images to same size
            target_size = (512, 512)
            char_resized = character_img.resize(target_size, Image.Resampling.LANCZOS)
            bg_resized = background_img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy arrays for processing
            char_array = np.array(char_resized)
            bg_array = np.array(bg_resized)
            
            # Create a simple mask based on brightness (this is a basic approach)
            # In a more advanced version, you'd use proper image segmentation
            char_gray = cv2.cvtColor(char_array, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(char_gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Convert mask to 3 channels
            mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
            
            # Blend images using the mask
            blended = (char_array * mask_3d + bg_array * (1 - mask_3d)).astype(np.uint8)
            
            # Convert back to PIL Image
            combined_image = Image.fromarray(blended)
            
            logger.info("Images combined successfully")
            return combined_image
            
        except Exception as e:
            logger.error(f"Error combining images: {e}")
            return self._side_by_side_combination(character_img, background_img)
    
    def _side_by_side_combination(self, char_img, bg_img):
        """Fallback: combine images side by side"""
        try:
            # Resize both images
            size = (256, 512)
            char_resized = char_img.resize(size, Image.Resampling.LANCZOS)
            bg_resized = bg_img.resize(size, Image.Resampling.LANCZOS)
            
            # Create combined image
            combined = Image.new('RGB', (512, 512))
            combined.paste(char_resized, (0, 0))
            combined.paste(bg_resized, (256, 0))
            
            # Add a subtle border
            from PIL import ImageDraw
            draw = ImageDraw.Draw(combined)
            draw.line([(256, 0), (256, 512)], fill=(255, 255, 255), width=2)
            
            logger.info("Side-by-side combination created")
            return combined
            
        except Exception as e:
            logger.error(f"Error in side-by-side combination: {e}")
            return self._create_placeholder_image(512, 512, "Combined")
    
    def _create_placeholder_image(self, width, height, text):
        """Create a high-quality placeholder image"""
        from PIL import ImageDraw, ImageFont
        
        # Create gradient background
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        # Create a nice gradient
        for y in range(height):
            r = int(100 + (y / height) * 100)
            g = int(150 + (y / height) * 50)
            b = int(200 + (y / height) * 55)
            color = (min(r, 255), min(g, 255), min(b, 255))
            draw.line([(0, y), (width, y)], fill=color)
        
        # Add decorative elements
        # Draw some geometric shapes
        draw.ellipse([width//4, height//4, 3*width//4, 3*height//4], 
                    outline=(255, 255, 255), width=3)
        draw.rectangle([width//3, height//3, 2*width//3, 2*height//3], 
                      outline=(255, 255, 255), width=2)
        
        # Add text
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text with shadow
        draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0, 128))
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        
        logger.info(f"Placeholder image created: {text}")
        return image
    
    def save_image(self, image, filename):
        """Save image to media directory"""
        try:
            media_path = os.path.join(settings.MEDIA_ROOT, 'generated_images')
            os.makedirs(media_path, exist_ok=True)
            
            filepath = os.path.join(media_path, filename)
            image.save(filepath, 'JPEG', quality=90, optimize=True)
            
            logger.info(f"Image saved: {filename}")
            return os.path.join('generated_images', filename)
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None
    
    def cleanup_models(self):
        """Clean up models to free memory"""
        if hasattr(self, 'pipe') and self.pipe:
            del self.pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("Models cleaned up")