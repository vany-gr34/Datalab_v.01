graph TD
    A[User/Application] --> B[ingest() Function]
    B --> C[IngestionRegistry]
    C --> D{Source Type}
    D -->|file| E[FileIngestor]
    D -->|db| F[DatabaseIngestor]
    D -->|api| G[APIIngestor]
    D -->|upload| H[UserUploadIngestor]
    E --> I[BaseIngestor.ingest()]
    F --> I
    G --> I
    H --> I
    I --> J[Dataset Object]
    J --> K[Data: pandas.DataFrame]
    J --> L[Metadata: Dict]
    C --> M[RawDataStorage]
    C --> N[MetadataManager]
    C --> O[IngestionLogger]
    M --> P[Store Raw Data]
    N --> Q[Save Metadata]
    O --> R[Log Events]
    J --> S[Preprocessing Layer]
    S --> T[Modeling Layer]
    T --> U[Visualization Layer]
    U --> V[Deployment Layer]

    classDef component fill:#e1f5fe
    classDef data fill:#f3e5f5
    classDef storage fill:#e8f5e8

    class A,B component
    class J,K,L data
    class M,N,O,P,Q,R storage
