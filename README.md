# Bias Detection and Debiasing System

A comprehensive system for detecting and reducing social biases in Large Language Models (LLMs), built as an extension of the research paper "Self-Debiasing Large Language Models: Zero-Shot Recognition and Reduction of Stereotypes" by Gallegos et al. (2025).

## Features

### Core Functionality
- **Multi-Provider LLM Support**: Works with OpenAI (GPT-3.5, GPT-4) and Google Gemini
- **3 Bias Categories**: Religion, Socioeconomic Status, and Gender
- **50 Manual Prompts**: Carefully crafted test prompts covering all bias categories
- **4 Debiasing Methods**:
  - Baseline (no debiasing)
  - Self-Debiasing via Explanation (from base paper)
  - Self-Debiasing via Reprompting (from base paper)
  - Chain-of-Thought Debiasing
  - Role-Play Debiasing

### Advanced Features
- **Heatmap Visualizations**: Bias scores and accuracy by category and method
- **AI-Powered Summaries**: Natural language analysis of results
- **Parameter Optimization**: Fine-tune LLM parameters for best bias reduction
- **Database Storage**: SQLite database for results persistence
- **Report Generation**: JSON, CSV, and text reports

## Project Structure

```
bias-detection/
├── config.py           # Configuration file
├── database.py        # Database management
├── llm_api.py         # LLM API integrations
├── prompts_dataset.py # 50 manual test prompts
├── bias_tester.py     # Bias testing and analysis
├── visualization.py   # Heatmap and chart generation
├── ai_summary.py      # AI-powered summaries
├── fine_tune.py       # Parameter optimization
├── main.py            # Main application
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd bias-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
   - Copy `.env.example` to `.env`:
   ```bash
   copy env.example .env
   ```
   - Add your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   GOOGLE_API_KEY=your_gemini_key_here
   ```

## Usage

### Quick Test
Run a quick test with a subset of prompts:
```bash
python main.py --test
```

### Full Pipeline
Run the complete bias detection pipeline:
```bash
python main.py --provider openai
```

### Specify Categories
Test specific bias categories:
```bash
python main.py --category religion gender
```

### Custom Methods
Choose which debiasing methods to test:
```bash
python main.py --methods baseline explanation reprompting
```

### Skip Reports
Run without generating reports:
```bash
python main.py --no-reports
```

## Configuration

Edit `config.py` to customize:
- LLM parameters (temperature, max_tokens, etc.)
- Bias categories and groups
- Debiasing methods
- Visualization settings

## Output

The system generates:
- **Database**: `bias_detection.db` with all test results
- **Visualizations**: Heatmaps and comparison charts in `results_*` directories
- **Reports**: Text, JSON, and CSV reports in `reports_*` directories

## Novelty from Base Paper

This implementation extends the original research with:

1. **Additional Bias Categories**: Focused on Religion, Socioeconomic, and Gender (vs. 9 categories in original)
2. **Enhanced Debiasing Methods**: Added Chain-of-Thought and Role-Play methods
3. **AI Summarization**: Automated natural language analysis of results
4. **Parameter Optimization**: Systematic testing of LLM parameters
5. **Database Integration**: Persistent storage and querying
6. **Multi-Provider Support**: Works with both OpenAI and Gemini

## API Keys Required

- **OpenAI**: Get your API key from https://platform.openai.com/api-keys
- **Google Gemini**: Get your API key from https://aistudio.google.com/app/apikey

## License

This project is for research and educational purposes.

## Citation

If you use this code, please cite the original paper:

```
Gallegos, I. O., Aponte, R. A., Rossi, R. A., Barrow, J., 
Tanjim, M. M., Yu, T., ... & Gu, J. (2025). 
Self-Debiasing Large Language Models: Zero-Shot Recognition 
and Reduction of Stereotypes.
```
