# Enterprise Copilot Specification

## Overview

The system provides grounded answers with source citations.

## Core Capabilities

- Hybrid dense and sparse search
- Reciprocal rank fusion
- Citation verification agent

```python
def query_copilot(prompt: str) -> str:
    return 'Answer with citations'
```

### Limitations

Scanned documents require future OCR capabilities.