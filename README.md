# DataLab v0.1 - Automated ML Workflow Platform

An interactive machine learning platform that automates the entire ML workflow for researchers and non-data scientists. Built with Streamlit, this platform provides an intuitive interface for data ingestion, preprocessing, model training, and deployment.

## 🚀 Features

### Data Ingestion Layer
- **Multi-Source Support**: Ingest data from files (CSV, Excel, JSON, Parquet), databases (PostgreSQL, MySQL, SQLite), and REST APIs
- **User Upload Interface**: Streamlit-based file upload with support for multiple formats
- **Metadata Management**: Automatic metadata generation and dataset tracking
- **Error Handling**: Comprehensive error handling and logging throughout the ingestion pipeline
- **Pluggable Architecture**: Easily extensible for new data sources

### Machine Learning Pipeline
- **Automated Preprocessing**: Handle missing values, scaling, encoding, and feature engineering
- **Model Registry**: Support for multiple classification and regression algorithms
- **Model Training**: Automated hyperparameter tuning and cross-validation
- **Visualization**: Interactive charts and plots for data exploration
- **Deployment Ready**: Export trained models for production use

## 🏗️ Architecture

### Data Ingestion Layer Architecture

```
ingestion/
├── __init__.py              # Main ingestion interface
├── base_ingestor.py         # Abstract base class for all ingestors
├── file_ingestor.py         # Local file ingestion (CSV, Excel, JSON, Parquet)
├── db_ingestor.py           # Database ingestion (PostgreSQL, MySQL, SQLite)
├── api_ingestor.py          # REST API ingestion with authentication
├── user_upload_ingestor.py  # User upload handling via Streamlit
├── metadata.py              # Metadata generation and management
├── storage.py               # Raw data storage strategies
├── logger.py                # Centralized logging
└── ingestion_registry.py    # Pluggable module management
```

### Core Components

#### Unified Ingestion Interface
```python
from ingestion import ingest

# File ingestion
dataset = ingest('file', {'file_path': 'data.csv'})

# User upload ingestion
dataset = ingest('upload', {
    'file_content': base64_content,
    'file_name': 'uploaded.csv',
    'file_type': 'csv'
})

# Database ingestion
dataset = ingest('database', {
    'connection_string': 'postgresql://user:pass@localhost/db',
    'query': 'SELECT * FROM table'
})

# API ingestion
dataset = ingest('api', {
    'url': 'https://api.example.com/data',
    'auth': {'type': 'bearer', 'token': 'your-token'}
})
```

#### Dataset Object
Each ingestion returns a standardized `Dataset` object:
```python
@dataclass
class Dataset:
    id: str                    # Unique dataset identifier
    data: pd.DataFrame         # The actual data
    metadata: Dict            # Metadata about the dataset
    source_type: str          # Type of data source
    created_at: datetime      # Creation timestamp
```

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd datalab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 🔧 Configuration

### Database Connections
Configure database connections in your environment or configuration file:
```python
# PostgreSQL
connection_string = "postgresql://user:password@localhost:5432/database"

# MySQL
connection_string = "mysql://user:password@localhost:3306/database"

# SQLite
connection_string = "sqlite:///database.db"
```

### API Authentication
Supported authentication methods:
- **None**: Public APIs
- **API Key**: Header-based authentication
- **Bearer Token**: JWT token authentication
- **Basic Auth**: Username/password authentication

## 📊 Usage Examples

### File Upload via Streamlit
```python
uploaded_file = st.file_uploader("Upload data file", type=["csv", "xlsx", "json"])

if uploaded_file:
    # Convert to base64 for ingestion layer
    file_content = base64.b64encode(uploaded_file.read()).decode()

    # Determine file type
    file_type = uploaded_file.name.split('.')[-1].lower()

    # Ingest the data
    dataset = ingest('upload', {
        'file_content': file_content,
        'file_name': uploaded_file.name,
        'file_type': file_type
    })

    st.success(f"Successfully ingested {uploaded_file.name}!")
    st.dataframe(dataset.data.head())
```

### Database Ingestion
```python
# Connect to PostgreSQL database
dataset = ingest('database', {
    'connection_string': 'postgresql://user:pass@localhost/db',
    'table': 'customers',
    'limit': 1000
})
```

### API Data Fetching
```python
# Fetch data from REST API
dataset = ingest('api', {
    'url': 'https://jsonplaceholder.typicode.com/posts',
    'method': 'GET',
    'headers': {'Authorization': 'Bearer your-token'}
})
```

## 🔒 Security Features

- **Input Sanitization**: All inputs are validated and sanitized
- **File Size Limits**: Configurable file size limits for uploads
- **SQL Injection Prevention**: Parameterized queries for database operations
- **Rate Limiting**: Built-in rate limiting for API calls
- **Error Handling**: Comprehensive error handling without exposing sensitive information

## 🚀 Scalability Features

- **Chunked Processing**: Large files are processed in chunks
- **Async Support**: Asynchronous processing for long-running operations
- **Memory Management**: Efficient memory usage for large datasets
- **Caching**: Intelligent caching of processed datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data processing powered by [pandas](https://pandas.pydata.org/)
- Machine learning with [scikit-learn](https://scikit-learn.org/)

---

**Note**: This is an educational project aimed at simplifying the machine learning workflow for researchers and students. For production use, additional security measures and performance optimizations may be required.
