# Drone Review Analysis with Ollama

This code uses qwen2.5coder LLM models of 3 different sizes to analyze reviews of drones.

## Features

- **Multi-Model Analysis**: Compare results across different Ollama models
- **Technical Term Recognition**: Identifies drone-specific terminology in Russian
- **Sentiment Analysis**: Classifies reviews as positive, negative, or neutral
- **Topic Extraction**: Automatically detects main topics and issues
- **Quality Evaluation**: Comprehensive model performance comparison
- **Batch Processing**: Analyze multiple reviews efficiently

## Models Supported

The system is configured to work with these Ollama models:
- `qwen2.5-coder:7b`
- `qwen2.5-coder:3b` 
- `qwen2.5-coder:1.5b`

You can easily modify the `model_names` list to use different Ollama models.

## Installation

### Prerequisites

- Python 3.7+
- Ollama installed and running on localhost:11434
- Required Ollama models pulled and available

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd drone-review-analysis
```

2. Install required dependencies:
```bash
pip install requests pandas numpy scikit-learn
```

3. Ensure Ollama is running:
```bash
ollama serve
```

4. Pull the required models:
```bash
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5-coder:1.5b
```

## Usage

### Basic Analysis

1. Prepare your reviews in a JSON file (`reviews.json`):
```json
[
  "Great drone with excellent camera quality but short battery life",
  "Poor build quality and GPS issues from day one",
  "Average performance for the price point"
]
```

2. Run the analysis:
```bash
python drone_analyzer.py
```

### Customizing Analysis

Modify the `model_names` list in the `main()` function to use different models:

```python
model_names = [
    "your-preferred-model:7b",
    "another-model:3b"
]
```

### Output Files

The analysis generates:
- `drone_reviews_ollama_analysis_YYYYMMDD_HHMMSS.json`: Complete analysis results
- `drone_reviews_comparison_YYYYMMDD_HHMMSS.csv`: Comparison table for easy review

## Evaluation Metrics

The system evaluates models on:

- **Sentiment Agreement**: Percentage of reviews where models agree on sentiment
- **Technical Term Understanding**: Ability to recognize drone-specific terminology
- **Rating Consistency**: Variance in numerical ratings (lower is better)
- **Response Quality**: Overall quality of JSON responses and field completion

## Technical Details

### Analysis Fields

For each review, the system extracts:
- `sentiment`: Overall tone (positive/negative/neutral)
- `main_topic`: Primary discussion topic
- `issue`: Specific problem or advantage mentioned
- `rating`: Numerical score from 1-5

### Supported Technical Terms

The system recognizes 35+ Russian drone-related terms including:
- Battery/charging terms: `батарея`, `аккумулятор`, `время полета`
- Camera terms: `камера`, `стабилизация`, `4k`, `автофокус`
- Flight terms: `gps`, `навигация`, `полет`, `маневренность`
- Hardware terms: `пропеллер`, `мотор`, `датчик`, `контроллер`

## Configuration

### Model Parameters

Current configuration uses:
- Temperature: 0.3 (for consistent results)
- Top-p: 0.9
- Timeout: 120 seconds
- Retry attempts: 3

### Customization

You can modify:
- Technical terms list in `technical_terms`
- Prompt structure in `create_prompt()`
- Evaluation metrics in `ModelEvaluator` class
- Model parameters in `analyze_with_model()`

## Example Output

```json
{
  "sentiment": "положительный",
  "main_topic": "качество камеры", 
  "issue": "отличное качество съемки",
  "rating": 4
}
```

## Troubleshooting

### Common Issues

1. **Ollama connection refused**:
   - Ensure Ollama is running: `ollama serve`
   - Check if port 11434 is accessible

2. **Model not found**:
   - Verify model names are correct
   - Pull models using `ollama pull <model-name>`

3. **JSON parsing errors**:
   - Models might return malformed JSON
   - System includes fallback responses for errors

4. **Timeout issues**:
   - Increase timeout in `analyze_with_model()`
   - Check system resources

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Disclaimer

This tool is designed for research and analysis purposes. The accuracy of sentiment analysis and topic extraction may vary depending on the quality of the language models and input data. Always verify critical results manually.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Ensure all prerequisites are met
3. Verify model compatibility with Ollama
4. Check the example reviews format

---

*Note: This tool is specifically optimized for Russian-language drone reviews but can be adapted for other languages and product categories.*
