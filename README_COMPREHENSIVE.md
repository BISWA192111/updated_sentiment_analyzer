
# � Medical Conversation Analyzer

**Advanced AI-Powered Medical Conversation Analysis Platform**

A comprehensive, production-ready application for analyzing medical conversations with state-of-the-art AI models. Upload audio files to get detailed insights including transcription, patient sentiment analysis, treatment phase detection, and satisfaction scoring.

## ✨ Key Features

### 🎙️ **Advanced Transcription & Speaker Identification**
- **Whisper-based Speech Recognition** with automatic speaker diarization
- **Real-time Processing** of audio files up to 100MB
- **Multi-format Support**: WAV, MP3, M4A, FLAC
- **Speaker Classification**: Automatic doctor/patient identification

### 😊 **Comprehensive Sentiment Analysis**
- **XLM-RoBERTa + BiLSTM** contextual sentiment analysis
- **Patient-focused Analysis** with confidence scores
- **Multilingual Support** with automatic language detection
- **Contextual Understanding** using conversation history

### 📅 **Treatment Phase Detection**
- **Multi-layered Classification**: Before/During/After treatment phases
- **Keyword & Grammar Analysis** for precise phase identification
- **Medical Context Understanding** for accurate classification
- **Visual Phase Distribution** with interactive timeline

### 📊 **Patient Satisfaction Scoring**
- **Advanced Satisfaction Algorithm** with Temperature scaling and calibration
- **Sentiment-based Metrics** with dampened negative impact
- **Comprehensive Dashboard** with visual breakdowns
- **Real-time Satisfaction Level** assessment

### 🔒 **Privacy & Compliance**
- **HIPAA-compliant PHI Scrubbing** with audit logging
- **Local Processing** (no cloud uploads required)
- **Bias Detection & Mitigation** framework
- **Secure Temporary File Handling**

## 🚀 How to Use

1. **📁 Upload Audio**: Select your medical conversation audio file
2. **🔍 Analyze**: Click "Analyze Conversation" to start processing  
3. **📊 Review Results**: Explore the comprehensive analysis across four tabs:
   - **🎙️ Conversation Transcript**: Complete transcription with sentiment analysis
   - **� Patient Satisfaction**: Detailed satisfaction metrics and scoring
   - **📅 Treatment Phase Analysis**: Phase breakdown and distribution
   - **📋 Analysis Summary**: Key insights and quality metrics

## 🤖 AI Models & Technology

- **🎙️ Speech Recognition**: OpenAI Whisper (base/small/tiny models)
- **😊 Sentiment Analysis**: XLM-RoBERTa multilingual + BiLSTM contextual
- **👥 Speaker Diarization**: pyannote.audio 3.1 with DistilBERT classification
- **📅 Phase Detection**: Multi-layer keyword and grammar analysis
- **🧠 Contextual Understanding**: Advanced conversation history management
   - Downloads and processes the audio
   - Separates speakers (doctor vs patient)
   - Transcribes speech using Whisper
   - Applies HIPAA-compliant PHI scrubbing
   - Detects languages and translates if needed
   - Analyzes sentiment using advanced ML models
3. **Results**: Get comprehensive analysis including:
   - Patient satisfaction scores
   - Conversation timeline
   - Sentiment flow visualization
   - Detailed breakdown of each conversation turn

## 📊 Analysis Output

- **Summary Dashboard**: Overview of key metrics
- **Patient Satisfaction Score**: 0-100 scale with confidence levels
- **Conversation Timeline**: Turn-by-turn analysis
- **Sentiment Visualization**: Interactive charts showing sentiment flow
- **HIPAA Compliance Report**: PHI detection and scrubbing statistics

## 🔧 Technical Stack

- **Frontend**: Gradio for interactive web interface
- **Audio Processing**: Whisper (OpenAI), pyannote.audio, pydub
- **ML Models**: DistilBERT, Transformers, scikit-learn
- **Sentiment Analysis**: VADER, TextBlob, custom medical models
- **Speaker Diarization**: pyannote.audio pipeline
- **Language Processing**: NLTK, langdetect
- **Visualization**: Matplotlib, Seaborn, Plotly

## 🛡️ Privacy & Compliance

- **HIPAA Compliant**: Automatically removes PHI from transcripts
- **Data Security**: No audio files are permanently stored
- **Privacy First**: All processing is done in real-time
- **Audit Trail**: Comprehensive logging of PHI scrubbing activities

## 📝 Usage Examples

### Basic Analysis
Simply paste an audio URL and click "Analyze Audio" to get:
- Overall sentiment analysis
- Speaker identification
- Basic satisfaction metrics

### Comprehensive Analysis
Choose "Comprehensive" mode for:
- Detailed conversation flow
- Advanced sentiment scoring
- Phase-based analysis (before/during/after)
- Multi-model ensemble results

## 🔗 Example Audio URLs

You can test the system with these sample URLs:
- https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav
- https://file-examples.com/storage/fe86b2a7e15c6fc09a3cbf0/2017/11/file_example_WAV_1MG.wav

## 📋 Requirements

See `requirements.txt` for full dependency list. Key requirements:
- Python 3.9+
- PyTorch
- Transformers
- Gradio
- Audio processing libraries

## 🚀 Deployment

This app is designed for easy deployment on Hugging Face Spaces:

1. Clone this repository
2. Upload to Hugging Face Spaces
3. The app will automatically install dependencies and launch

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions or support, please open an issue in the repository.

---

**⚠️ Disclaimer**: This tool is for research and educational purposes. Always consult with healthcare professionals for medical decisions.

