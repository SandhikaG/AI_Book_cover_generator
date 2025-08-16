#!/usr/bin/env python3
"""
Book Cover Generator - Gradio Interface
DreamBooth trained Stable Diffusion model for generating book covers
"""

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import numpy as np

# Global pipeline variable
pipeline = None

def load_model():
    """Load the DreamBooth trained model"""
    global pipeline
    
    try:
        print("Loading DreamBooth model...")
        
        # Try to load from local model directory
        model_path = "./model"
        if os.path.exists(model_path):
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
        else:
            # Fallback to base model if custom model not found
            print("Custom model not found, using base model...")
            pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")
            
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def add_text_to_cover(image, title, author="", subtitle=""):
    """Add text overlay to the generated book cover"""
    
    if image is None:
        return None
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Create a copy to work with
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    W, H = img.size
    
    # Try to load a decent font, fallback to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        author_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 30)
    except:
        try:
            title_font = ImageFont.load_default()
            author_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
        except:
            # Skip text overlay if font loading fails completely
            return img
    
    # Simple text positioning
    title_y = H // 4
    subtitle_y = title_y + 80
    author_y = H - 100
    
    # Add shadow effect
    def draw_text_with_shadow(pos, text, font, fill_color="white", shadow_color="black"):
        x, y = pos
        # Draw shadow
        for adj in range(1, 4):
            draw.text((x-adj, y-adj), text, font=font, fill=shadow_color)
            draw.text((x+adj, y+adj), text, font=font, fill=shadow_color)
        # Draw main text
        draw.text(pos, text, font=font, fill=fill_color)
    
    # Draw title
    if title:
        title_width = draw.textlength(title, font=title_font) if hasattr(draw, 'textlength') else len(title) * 30
        title_x = (W - title_width) // 2
        draw_text_with_shadow((title_x, title_y), title, title_font, "white", "black")
    
    # Draw subtitle
    if subtitle:
        subtitle_width = draw.textlength(subtitle, font=subtitle_font) if hasattr(draw, 'textlength') else len(subtitle) * 20
        subtitle_x = (W - subtitle_width) // 2
        draw_text_with_shadow((subtitle_x, subtitle_y), subtitle, subtitle_font, "#f0f0f0", "black")
    
    # Draw author
    if author:
        author_width = draw.textlength(author, font=author_font) if hasattr(draw, 'textlength') else len(author) * 25
        author_x = (W - author_width) // 2
        draw_text_with_shadow((author_x, author_y), f"by {author}", author_font, "#e0e0e0", "black")
    
    return img

def generate_book_cover(title, plot_description, author="", subtitle="", 
                       num_inference_steps=50, guidance_scale=7.5, add_text=True):
    """Generate a book cover using the DreamBooth model"""
    
    global pipeline
    
    if pipeline is None:
        return None, "❌ Model not loaded. Please wait for initialization."
    
    if not title or not plot_description:
        return None, "❌ Please provide both title and plot description."
    
    try:
        # Create prompt with DreamBooth trigger
        prompt = f"a sks book cover illustration of {plot_description}, highly detailed, professional book cover art, no text, no words"
        
        # Negative prompt to avoid unwanted text
        negative_prompt = "text, words, letters, title, subtitle, author name, typography, writing, blurry, low quality"
        
        print(f"Generating: {prompt}")
        
        # Generate image
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=768,
            width=512
        )
        
        image = result.images[0]
        
        # Add text overlay if requested
        if add_text:
            image = add_text_to_cover(image, title, author, subtitle)
        
        return image, f"✅ Book cover generated successfully!"
        
    except Exception as e:
        error_msg = f"❌ Error generating cover: {str(e)}"
        print(error_msg)
        return None, error_msg

def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="📚 AI Book Cover Generator", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 📚 AI Book Cover Generator
        
        Generate professional book covers using AI! This tool uses a custom-trained DreamBooth model 
        specifically fine-tuned for creating book cover artwork.
        
        **Instructions:**
        1. Enter your book title and plot description
        2. Optionally add author name and subtitle  
        3. Adjust generation parameters if needed
        4. Click "Generate Book Cover"
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📖 Book Information")
                
                title_input = gr.Textbox(
                    label="Book Title",
                    placeholder="Enter the book title...",
                    value="The Dragon's Quest"
                )
                
                plot_input = gr.Textbox(
                    label="Plot Description", 
                    placeholder="Describe the story, genre, setting, mood...",
                    lines=3,
                    value="epic fantasy adventure with dragons, magical castles, and brave heroes"
                )
                
                author_input = gr.Textbox(
                    label="Author Name (Optional)",
                    placeholder="Author's name...",
                    value=""
                )
                
                subtitle_input = gr.Textbox(
                    label="Subtitle (Optional)",
                    placeholder="Book subtitle...", 
                    value=""
                )
                
                gr.Markdown("### ⚙️ Generation Settings")
                
                add_text_checkbox = gr.Checkbox(
                    label="Add text overlay to cover",
                    value=True
                )
                
                steps_slider = gr.Slider(
                    minimum=20,
                    maximum=100, 
                    value=50,
                    step=5,
                    label="Inference Steps (Higher = Better Quality, Slower)"
                )
                
                guidance_slider = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.5,
                    label="Guidance Scale (Higher = More Prompt Following)"
                )
                
                generate_btn = gr.Button("🎨 Generate Book Cover", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ Generated Cover")
                
                output_image = gr.Image(
                    label="Your Book Cover",
                    type="pil",
                    height=600
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready to generate...",
                    interactive=False
                )
        
        # Example gallery
        gr.Markdown("### 📚 Example Book Covers")
        
        examples = [
            ["The Cosmic Journey", "space adventure with distant planets and starships", "Dr. Cosmos", "", 50, 7.5, True],
            ["Haunted Manor", "gothic horror mystery in an old Victorian mansion", "E. A. Mystery", "A Spine-Chilling Tale", 50, 7.5, True],
            ["Cyberpunk City", "futuristic cyberpunk thriller in neon-lit metropolis", "Neo Author", "", 50, 7.5, True],
            ["The Enchanted Forest", "magical fantasy adventure through mystical woodlands", "Fantasy Writer", "Book One", 50, 7.5, True]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[title_input, plot_input, author_input, subtitle_input, steps_slider, guidance_slider, add_text_checkbox],
            outputs=[output_image, status_text],
            fn=generate_book_cover,
            cache_examples=False
        )
        
        # Connect the generate button
        generate_btn.click(
            fn=generate_book_cover,
            inputs=[title_input, plot_input, author_input, subtitle_input, steps_slider, guidance_slider, add_text_checkbox],
            outputs=[output_image, status_text]
        )
        
        gr.Markdown("""
        ---
        ### 💡 Tips for Better Results:
        - Be specific about genre, mood, and visual elements in your plot description
        - Try different guidance scale values (7.5 is usually good)
        - Higher inference steps = better quality but slower generation
        - The model works best with fantasy, sci-fi, mystery, and adventure themes
        
        ### 🔧 Technical Info:
        - Model: Custom DreamBooth trained on book cover dataset
        - Trigger phrase: "sks book cover" (automatically added)
        - Base resolution: 512x768 (book cover aspect ratio)
        """)
    
    return demo

def main():
    """Main function to run the Gradio app"""
    
    print("🚀 Starting Book Cover Generator...")
    
    # Load the model
    if not load_model():
        print("❌ Failed to load model. Please check your model files.")
        return
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    print("✅ Gradio interface ready!")
    print("🌐 Launching web interface...")
    
    # Launch with public link for sharing
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public link
        debug=True
    )

if __name__ == "__main__":
    main()