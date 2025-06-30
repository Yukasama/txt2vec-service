# Vectorize - Text Embedding Service

Welcome to **Vectorize**, a Text Embedding Service developed for Robert Bosch GmbH in the AWP Module at Hochschule Karlsruhe. This comprehensive platform enables organizations to manage the complete lifecycle of text embedding workflows with production-ready reliability and scalability.

## 🚀 What is Vectorize?

**Vectorize** is a self-hosted text embedding service that simplifies the process of building, training, and deploying custom embedding models. From corpus upload to model evaluation, Vectorize provides a unified platform for NLP experimentation and production deployment.

### Core Capabilities

- **📊 Dataset Management**: Upload, validate, and process training datasets from multiple sources
- **🤖 Model Training**: Train custom embedding models tailored to your specific domain
- **📈 Evaluation Framework**: Comprehensive model evaluation with industry-standard metrics
- **🔄 Synthetic Data Generation**: Generate high-quality synthetic datasets for model improvement
- **⚡ Background Processing**: Async task execution with real-time monitoring and status tracking
- **🔌 RESTful API**: Complete API for seamless integration with existing workflows
- **🐳 Production Ready**: Docker containerization with enterprise deployment support

### Key Components

- **🌐 API Layer**: FastAPI-based REST endpoints with automatic OpenAPI documentation
- **⚙️ Service Layer**: Business logic orchestration and workflow management
- **🗄️ Repository Layer**: Data access patterns with SQLModel ORM
- **📋 Task System**: Asynchronous background processing with Dramatiq
- **🔧 Configuration**: Environment-based configuration with validation
- **🛠️ Utilities**: Shared components and helper functions

## 🚀 Quick Start

Get up and running with Vectorize in just a few steps:

1. **🔧 [Setup Environment](installation.md)** - Install dependencies and configure your development environment
2. **⚙️ [Configure Settings](configuration.md)** - Set up your `.env` file with required configurations
3. **▶️ [Start the Server](installation.md#running-vectorize)** - Launch Vectorize locally or with Docker
4. **📖 [Explore the API](api.md)** - Discover available endpoints and capabilities

### Quick Commands

```bash
# Clone and setup
git clone https://github.com/yukasama/vectorize.git
cd vectorize
uv sync
cp .env.example .env

# Start development server
uv run app

# Or with Docker
docker compose up
```

## 🏢 Project Structure

```
vectorize/
├── 📁 src/vectorize/         # Core application code
│   ├── 🤖 ai_model/          # AI model management and operations
│   ├── 🔧 common/            # Shared utilities and error handling
│   ├── ⚙️ config/            # Configuration management system
│   ├── 📊 dataset/           # Dataset upload and processing
│   ├── 📈 evaluation/        # Model evaluation framework
│   ├── 🔮 inference/         # Model inference endpoints
│   ├── 🔄 synthesis/         # Synthetic data generation
│   ├── 📋 task/              # Background task orchestration
│   ├── 🎯 training/          # Model training workflows
│   ├── 📤 upload/            # Multi-source upload handling
│   └── 🛠️ utils/             # Shared utility functions
├── 🧪 tests/                 # Comprehensive test suite
├── 📚 docs/                  # Documentation and guides
├── 🔨 scripts/               # Development and deployment scripts
└── 📋 resources/             # Configuration files and assets
```

## 📚 Documentation Guide

### 🏁 Getting Started

| Guide                                         | Description                                         |
| --------------------------------------------- | --------------------------------------------------- |
| [📦 Installation](installation.md)            | Complete setup guide for development and production |
| [⚙️ Configuration](configuration.md)          | Environment variables and settings management       |
| [🚀 Quick Start](installation.md#quick-start) | Get running in 5 minutes                            |

### 👥 User Guides

| Feature | Guide                                         | Description                                      |
| ------- | --------------------------------------------- | ------------------------------------------------ |
| 📊      | [Dataset Management](guides/datasets.md) | Upload, validate, and manage training datasets   |
| 🤖      | [AI Models](guides/models.md)            | Work with embedding models from multiple sources |
| 🎯      | [Model Training](guides/training.md)     | Train custom embedding models                    |
| 📈      | [Model Evaluation](guides/evaluation.md) | Evaluate and benchmark model performance         |
| 🔄      | [Synthetic Data](guides/synthesis.md)    | Generate synthetic datasets                      |
| 📋      | [Background Tasks](guides/tasks.md)      | Monitor and manage async operations              |

### 🔧 Developer Resources

| Resource                           | Description                          |
| ---------------------------------- | ------------------------------------ |
| [🔌 API Reference](api.md)         | Complete REST API documentation      |
| [🤝 Contributing](contributing.md) | How to contribute to the project     |
| [🎭 Local CI with Act](act.md)     | Run GitHub Actions workflows locally |

## 🌟 Key Features in Detail

### 🔄 Multi-Source Model Upload

- **Hugging Face Hub**: Direct integration with HF model repository
- **GitHub Repositories**: Load models from public/private GitHub repos
- **Local Files**: Upload models from your local filesystem
- **ZIP Archives**: Support for compressed model bundles

### 📊 Advanced Dataset Management

- **Format Support**: CSV, JSON, JSONL, XML, Excel files
- **Schema Validation**: HuggingFace dataset compatibility checking
- **Batch Processing**: Handle large datasets efficiently
- **Data Quality**: Automatic validation and cleaning

### 🎯 Flexible Training Pipeline

- **Custom Models**: Train embedding models on your specific data
- **Hyperparameter Tuning**: Configurable training parameters
- **Progress Monitoring**: Real-time training progress tracking
- **Checkpoint Management**: Save and restore training states

### 📈 Comprehensive Evaluation

- **Multiple Metrics**: Precision, recall, F1-score, and more
- **Benchmark Datasets**: Test against standard evaluation sets
- **Comparative Analysis**: Compare multiple models side-by-side
- **Detailed Reports**: Generate comprehensive evaluation reports

## 🤝 Contributing

We welcome contributions from the community! Here's how to get involved:

1. **🍴 Fork the Repository** - Create your own fork to work on
2. **🌿 Create a Feature Branch** - Keep your changes organized
3. **✅ Add Tests** - Ensure your code is well-tested
4. **📝 Update Documentation** - Help others understand your changes
5. **🔄 Submit a Pull Request** - Share your improvements with the community

See our [Contributing Guide](contributing.md) for detailed instructions.

## 👥 Contributors

We're grateful to all the talented individuals who have contributed to Vectorize:

<table>
<tr>
  <td align="center">
    <a href="https://github.com/Dosto1ewski">
      <img src="https://avatars.githubusercontent.com/Dosto1ewski" width="80" style="border-radius: 50%;" alt="Anselm Böhm"/>
      <br />
      <sub><b>Anselm Böhm</b></sub>
    </a>
  </td>
  <td align="center">
    <a href="https://github.com/BtnCbn">
      <img src="https://avatars.githubusercontent.com/BtnCbn" width="80" style="border-radius: 50%;" alt="Botan Coban"/>
      <br />
      <sub><b>Botan Coban</b></sub>
    </a>
  </td>
  <td align="center">
    <a href="https://github.com/yukasama">
      <img src="https://avatars.githubusercontent.com/yukasama" width="80" style="border-radius: 50%;" alt="Yukasama"/>
      <br />
      <sub><b>Yukasama</b></sub>
    </a>
  </td>
  <td align="center">
    <a href="https://github.com/domoar">
      <img src="https://avatars.githubusercontent.com/domoar" width="80" style="border-radius: 50%;" alt="Manuel Dausmann"/>
      <br />
      <sub><b>Manuel Dausmann</b></sub>
    </a>
  </td>
  <td align="center">
    <a href="https://github.com/Yannjc">
      <img src="https://avatars.githubusercontent.com/Yannjc" width="80" style="border-radius: 50%;" alt="Yannic Jahnke"/>
      <br />
      <sub><b>Yannic Jahnke</b></sub>
    </a>
  </td>
</tr>
</table>

## 📄 License

This project is licensed under the **Apache License, Version 2.0** - a permissive open-source license that:

- ✅ Allows commercial use
- ✅ Permits modification and distribution
- ✅ Provides patent protection
- ✅ Requires proper attribution

For complete terms and conditions, see the [full license text](https://www.apache.org/licenses/LICENSE-2.0).

---

**Ready to get started?** Check out our [Installation Guide](installation.md) or dive into the [API Documentation](api.md)!
