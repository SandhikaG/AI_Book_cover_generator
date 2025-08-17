#  AI Book Cover Generator 

A comprehensive pipeline for training and deploying custom DreamBooth models to generate professional book covers using AI.


##  Quick Links

- **[Try the Live Demo →](./gradio_app/)** - Use the trained model instantly
- **[Train Your Own Model →](./training/)** - Custom training pipeline with dataset scraping
- **[View Examples →](./Samples/)** - Generated book covers showcase

---

##  Project Overview

This project provides a complete end-to-end solution for AI-powered book cover generation:

1. **Dataset Collection**: Automated scraping of book covers from OpenLibrary
2. **Custom Model Training**: DreamBooth fine-tuning of Stable Diffusion v1.5 
3. **Professional Application**: Ready-to-deploy Gradio interface with text overlay
4. **Easy Deployment**: Multiple deployment options for production use

##  Project Structure

```
ai-book-cover-generator/
├── README.md                     # This comprehensive guide
├── training/                     # Complete training pipeline
│   ├── dataset_scraping.ipynb   # OpenLibrary data collection
│   ├── bookCoverGen_Model.ipynb # DreamBooth fine-tuning
│   └── README.md                # Training documentation
├── gradio_app/                   # Production-ready application
│   ├── app.py                   # Main Gradio interface
│   ├── requirements.txt         # Application dependencies
│   ├── model/                   # Trained model storage
│   └── README.md                # App deployment guide
└── examples/                     # Generated cover samples
    ├── fantasy/
    ├── sci-fi/
    └── mystery/
```

---

##  Getting Started

### Option 1: Use Pre-trained Model (Fastest)

Perfect for immediate use with our pre-trained model:

```bash
git clone https://github.com/SandhikaG/AI_Book_cover_generator.git
cd ai-book-cover-generator/gradio_app/
pip install -r requirements.txt
python app.py
```

**Open in browser**: http://localhost:7860

### Option 2: Train Your Own Model (Recommended)

For custom datasets and specialized book genres:

1. **Dataset Collection**: Open `training/dataset_scraping.ipynb` in Google Colab
2. **Model Training**: Open `training/bookCoverGen_Model.ipynb` in Google Colab  
3. **Deploy Application**: Use trained model with the Gradio app

---

##  Training Pipeline

### Dataset Scraping (`dataset_scraping.ipynb`)

**Automated OpenLibrary Integration**
- Searches OpenLibrary API for high-quality book covers
- Downloads 200-500 diverse images across multiple genres
- Generates descriptive captions for each cover
- Filters by quality (minimum 400x600 resolution)
- Creates training-ready dataset structure

**Output**: `lora_dataset.zip` with images and captions folders

### Model Fine-tuning (`bookCoverGen_Model.ipynb`)

**DreamBooth Training Process**
- **Base Model**: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
- **Fine-tuning Target**: UNet diffusion model (text encoder and VAE remain frozen)
- **Trigger Words**: "sks book cover" for concept association
- **Training Steps**: 800-1200 with loss monitoring
- **Prior Preservation**: Maintains general book cover knowledge

**Key Features**:
- Selective parameter training (UNet only)
- Memory optimization with gradient checkpointing
- Mixed precision training (fp16) for efficiency
- Regularization with class images to prevent overfitting

**Hardware Requirements**:
- Google Colab Pro recommended
- Minimum 12GB GPU VRAM
- Training time: 2-4 hours

---

##  Gradio Application

### Features

**Core Functionality**
- **Custom AI Model**: DreamBooth-trained specifically for book covers
- **Professional Text Overlay**: Automatic title, author, and subtitle placement
- **Multi-Genre Support**: Fantasy, sci-fi, mystery, romance, horror, and more
- **Real-time Generation**: 10-30 seconds per cover
- **High Quality Output**: 512x768 standard book cover resolution

### How to Use

1. **Enter Book Details**:
   - Title (required)
   - Plot description (required) - be specific about genre, mood, setting
   - Author name (optional)
   - Subtitle (optional)

2. **Adjust Generation Settings**:
   - **Inference Steps**: 20-100 (higher = better quality, slower)
   - **Guidance Scale**: 5.0-15.0 (higher = follows prompt more closely)
   - **Add Text Toggle**: Enable/disable automatic text overlay

3. **Generate**: Click "Generate Book Cover" and wait for results

### Technical Specifications

- **Base Model**: Stable Diffusion v1.5 with DreamBooth fine-tuning
- **Trigger Integration**: Automatically adds "sks book cover" to prompts
- **Resolution**: 512x768 pixels (standard book cover ratio)
- **Interface**: Professional Gradio web UI
- **Dependencies**: PyTorch, Diffusers, PIL, Gradio

---

##  Model Performance

### Strengths

**Genre Expertise**:
- **Fantasy**: Epic adventures, magical elements, dragons
- **Sci-Fi**: Futuristic cities, space themes, technology
- **Mystery**: Dark, noir aesthetics, atmospheric tension
- **Romance**: Elegant styling, soft color palettes
- **Horror**: Gothic elements, atmospheric darkness

**Technical Capabilities**:
- Professional cover layouts and composition
- Artistic and painterly styles
- Genre-appropriate imagery and color schemes
- Consistent book cover aesthetics
- Proper aspect ratios and design principles

### Limitations

- Limited photorealistic character portraits
- Best results with descriptive, genre-specific prompts
- Requires "sks book cover" trigger for optimal performance
- May need multiple generations for perfect results

---

##  Technology Stack

**Training Infrastructure**:
- **DreamBooth**: Concept learning and fine-tuning
- **Diffusers**: Hugging Face diffusion models library
- **Accelerate**: Multi-GPU training optimization
- **OpenLibrary API**: Dataset source for book covers

**Application Stack**:
- **Stable Diffusion v1.5**: Base generative model
- **Gradio**: Web interface framework
- **PIL/Pillow**: Image processing and text overlay
- **PyTorch**: Deep learning framework

**Deployment Options**:
- **Local Development**: Direct Python execution
- **Hugging Face Spaces**: Cloud hosting
- **Docker**: Containerized deployment
- **Custom Servers**: Self-hosted solutions

---

##  Training Configuration

### Key Parameters

- **Resolution**: 512x512 pixels during training
- **Learning Rate**: 1e-6 with constant scheduler
- **Batch Size**: 1 (memory constraints)
- **Training Steps**: 800-1200 (adjustable based on dataset)
- **Prior Loss Weight**: 1.0 for knowledge preservation
- **Class Images**: 200 regularization images

### Optimization Features

- **Memory Management**: Gradient checkpointing and 8-bit Adam
- **Training Stability**: Mixed precision (fp16) training
- **Quality Control**: Loss monitoring and periodic validation
- **Overfitting Prevention**: Prior preservation and class images

---

##  Deployment Guide

### Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ai-book-cover-generator.git
cd ai-book-cover-generator

# Install dependencies
pip install -r gradio_app/requirements.txt

# Add your trained model to gradio_app/model/
# Run application
cd gradio_app && python app.py
```

### Production Deployment

**Hugging Face Spaces**: Upload to Spaces for public access
**Docker Container**: Use provided Dockerfile for containerization
**Custom Server**: Deploy on your infrastructure with GPU support

### Model Integration

1. **Training Complete**: Download model files from Colab
2. **Extract Model**: Place in `gradio_app/model/` directory
3. **Verify Structure**: Ensure `model_index.json` and component folders exist
4. **Test Locally**: Run app.py to verify integration
5. **Deploy**: Choose your preferred deployment method

---


##  Best Practices

### For Training

- **Dataset Quality**: Use 300-500 high-quality, diverse book covers
- **Caption Accuracy**: Write detailed, genre-specific descriptions
- **Training Monitoring**: Watch loss curves and generate test images
- **Resource Management**: Use Colab Pro for stable GPU access

### For Generation

- **Prompt Engineering**: Be specific about genre, mood, and visual elements
- **Parameter Tuning**: Adjust inference steps and guidance scale per genre
- **Multiple Attempts**: Generate several options for best results
- **Text Overlay**: Use built-in text features for professional finishing


---

##  Troubleshooting

### Training Issues

**Out of Memory**: Enable gradient checkpointing, reduce batch size
**Poor Quality**: Increase training steps, improve dataset quality
**Dataset Errors**: Verify folder structure and file permissions

### Application Issues

**Model Loading**: Check model files and directory structure
**Generation Fails**: Verify GPU availability and dependencies
**Poor Results**: Adjust generation parameters and prompt specificity

---

##  Acknowledgments

- **Stability AI** for Stable Diffusion foundation model
- **Google Research** for DreamBooth training technique  
- **Gradio Team** for the intuitive web interface framework
- **OpenLibrary** for providing comprehensive book cover dataset
- **Hugging Face** for model hosting and diffusion libraries
